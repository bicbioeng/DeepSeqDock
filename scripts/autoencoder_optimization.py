# Default autoencoder optimization - does not contain any dataset specific evaluations

import os
import json
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import encoder as ec

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB
from hpbandster.core.worker import Worker

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import tensorflow as tf

import matplotlib
matplotlib.use('Agg') # Must be before importing matploblib.pyplot or pylab (see https://stackoverflow.com/questions/4706451/how-to-save-a-figure-remotely-with-pylab/4706614#4706614)
import matplotlib.pyplot as plt

import hpbandster.visualization as hpvis
import re


parser = argparse.ArgumentParser(description='This script optimizes an autoencoder model using the BOHB algorithm '
                                             '(see https://automl.github.io/HpBandSter/build/html/optimizers/bohb.html) based'
                                             ' on validation data loss. After optimization, an autoencoder is built using the '
                                             'ideal configuration determined by the optimization, and a '
                                             'keras model, BOHB quality control graph, and ideal model configuration.'
                                             ' This function is designed to take as input csv files with sample rows and feature columns.')
parser.add_argument('-d', '--train_data', type=str, required=True,
                    help='Path to training data csv. Samples as rows and features as columns. Required.')
parser.add_argument('-v', '--valid_data', type=str, required=True,
                    help='Path to validation data csv. Samples as rows and features as columns. Required.')
parser.add_argument('-t', '--test_data', type=str, nargs="*",
                    help='List of paths separated by spaces to test data csv(s). '
                         'Samples as rows and features as columns. Optional.', default=[])
parser.add_argument('-o', '--output_directory', type=str,
                    help='Directory to which output will be written', default='output')
parser.add_argument('--min_budget', type=float, help='Min number of epochs for training', default=100)
parser.add_argument('--max_budget', type=float, help='Max number of epochs for training', default=2000)
parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer', default=20)
parser.add_argument('--debug', help='Mode using code in. Debug tests on one random configuration', default='False')
parser.add_argument('--nic_name', type=str, help='Which network interface to use for communication.', default='lo')
parser.add_argument('--scheduler', type=str, help='Learning rate scheduler. One of 1cycle, 1cycle2, exponential, power',
                    default="power")
parser.add_argument('--modeltype', type=str, help="autoencoder", default="autoencoder")
parser.add_argument('--datetime', type=str, help="Used internally to identify runs")

args = parser.parse_args()


class KerasWorker(Worker):
    def __init__(self, train_data, valid_data, scheduler, modeltype='autoencoder', **kwargs):
        super().__init__(**kwargs)

        self.train_data = Path(train_data)
        self.valid_data = Path(valid_data)
        self.modeltype = modeltype
        self.scheduler = scheduler


        # Load datasets
        self.x_train = pd.read_csv(train_data, index_col=0).astype('float32')
        self.x_valid = pd.read_csv(valid_data, index_col=0).astype('float32')

        # Order features, highest variance to lowest
        self.feature_order = self.x_train.var(axis=0).sort_values(ascending=False).index

    def compute(self, config, budget, *args, **kwargs):

        # trim features
        train = self.x_train.loc[:, self.feature_order[0:config['nfeatures']]]
        valid = self.x_valid.loc[:, self.feature_order[0:config['nfeatures']]]

        if self.modeltype == 'autoencoder':
            model = ec.build_autoencoder(train=train, valid=valid,
                                         config=config)

        model.compile(loss=tf.keras.losses.mean_squared_error,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=config['lr']))
        
        if self.scheduler == '1cycle':
            onecycle_cb = ec.OneCycleLearningRate(np.ceil(train.shape[0] / config['batchsize']) * int(budget),
                                                      max_rate=config['lr'])
            callbacks = [onecycle_cb]
            
        elif self.scheduler == '1cycle2':
            lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
            ec.onecycle_lr(config['lr'], int(budget), 10, model=model))
            momentum_scheduler = ec.OneCycleMomentumCallback(int(budget), 10)
            callbacks = [lr_scheduler, momentum_scheduler]
            
        elif self.scheduler == 'exponential':
            lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
            ec.exponential_decay(lr0=config['lr'], s=config['s']))
            callbacks = [lr_scheduler]
            
        elif self.scheduler == 'power':
            lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
            ec.decayed_learning_rate(lr0=config['lr'], s=config['s']))
            callbacks = [lr_scheduler]

        model.fit(train, train,
                  batch_size=config['batchsize'],
                  epochs=int(budget),
                  verbose=0,
                  callbacks=callbacks,
                  validation_data=(valid, valid))

        # mean loss since loss of model is the sum of all individual
        train_score = model.evaluate(train, train, verbose=0)
        valid_score = model.evaluate(valid, valid, verbose=0) 

        if self.modeltype == 'autoencoder':
            # determine latent space for valid dataset to calculate metrics
            latent = tf.keras.Model(inputs=[model.layers[0].input], outputs=[model.layers[0].output])
            val = latent.predict(valid)

        # Metric dict
        metrics = {
            'valid loss': valid_score,
            'train loss': train_score,
        }
        return ({
            'loss': metrics['valid loss'],
            'info': metrics
        })

    @staticmethod
    def get_configspace(ntotal=10000):
        """Build configuration space
        :return: ConfigurationsSpace-Object
        """
        cs = CS.ConfigurationSpace()

        lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-1, default_value='0.005', log=True)

        dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0.0, upper=0.98, default_value=0.25,
                                                      log=False)
        noise = CSH.CategoricalHyperparameter('noise', [True, False])
        noise_std = CSH.UniformFloatHyperparameter('noise_std', lower=0.2, upper=1.0, default_value=0.5, log=False)

        cs.add_hyperparameters([lr, dropout_rate, noise, noise_std])
        cond = CS.EqualsCondition(noise_std, noise, True)
        cs.add_condition(cond)

        ntotal = np.clip(ntotal, a_min=0, a_max=10000)

        nfeatures = CSH.UniformIntegerHyperparameter('nfeatures', lower=250, upper=ntotal, default_value=np.ceil(ntotal/2))
        nlatent = CSH.UniformIntegerHyperparameter('nlatent', lower=16, upper=250, default_value=150)
        nlayers = CSH.UniformIntegerHyperparameter('nlayers', lower=2, upper=4, default_value=2)
        cs.add_hyperparameters([nfeatures, nlatent, nlayers])

        batchnorm = CSH.CategoricalHyperparameter('batchnorm', [True, False])
        l2reg = CSH.UniformFloatHyperparameter('l2reg', lower=1e-6, upper=1e-3, default_value=1e-5, log=True)
        batchsize = CSH.UniformIntegerHyperparameter('batchsize', lower=128, upper=1024, default_value=450, log=False)
        cs.add_hyperparameters([batchnorm, l2reg, batchsize])

        s = CSH.UniformIntegerHyperparameter('s', lower=1, upper=100, default_value=60)
        cs.add_hyperparameter(s)

        return cs


# set up directory for results - need to have ec file in same working directory as this file to run properly
if args.datetime:
    run_id = ("autoencoder_" + args.datetime)
else:
    run_id = time.strftime("autoencoder_%Y_%m_%d-%H_%M_%S")
    
if args.output_directory == "output":
    args.output_directory = Path.cwd() / 'output'
else:
    args.output_directory = Path(args.output_directory)

modelpath = args.output_directory / 'Raw Python Package' / 'Autoencoder' / run_id
if not os.path.exists(modelpath): modelpath.mkdir(parents=True)

qc_path = args.output_directory / 'Quality Assessment' / 'Autoencoder' / run_id
if not os.path.exists(qc_path): qc_path.mkdir(parents=True)

dataset_path = args.output_directory / 'Data Representations' / 'Autoencoder' / run_id
if not os.path.exists(dataset_path): dataset_path.mkdir(parents=True)

print("Autoencoder Run: ", run_id)

if args.debug == "True":
    worker = KerasWorker(run_id='0', train_data=args.train_data,
                         valid_data=args.valid_data,
                         modeltype=args.modeltype,
                         scheduler=args.scheduler)
    cs = worker.get_configspace()

    config = cs.sample_configuration().get_dictionary()
    print(config)
    res = worker.compute(config=config, budget=1, working_directory='.')
    print(res)

elif args.debug == "False":

    host = hpns.nic_name_to_host(args.nic_name)
    NS = hpns.NameServer(run_id=run_id, host=host, port=0,
                         working_directory=qc_path)
    ns_host, ns_port = NS.start()

    # Start local worker
    w = KerasWorker(train_data=args.train_data,
                    valid_data=args.valid_data,
                    scheduler=args.scheduler,
                    modeltype=args.modeltype,
                    run_id=run_id,
                    host=host,
                    nameserver=ns_host,
                    nameserver_port=ns_port)
    w.run(background=True)

    # Get number of features to give to configspace
    x_train = pd.read_csv(args.train_data, index_col=0).astype(
        'float32')

    #
    bohb = BOHB(configspace=w.get_configspace(ntotal=x_train.shape[1]),
                run_id=run_id, nameserver=ns_host,
                nameserver_port=ns_port,
                min_budget=args.min_budget, max_budget=args.max_budget)
    res = bohb.run(n_iterations=args.n_iterations)

    # Shut down namespace and workers so can be reused
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    # Make plots for optimization run
    all_runs = res.get_all_runs()
    curves = res.get_learning_curves()
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()

    hpvis.performance_histogram_model_vs_random(all_runs, id2config)
    plt.savefig(qc_path / 'histogram_model_vs_random.png',
                bbox_inches="tight", pad_inches=1.)

    hpvis.correlation_across_budgets(res)
    plt.savefig(qc_path / 'correlation_across_budgets.png',
                bbox_inches="tight", pad_inches=1.)

    hpvis.losses_over_time(all_runs)
    plt.savefig(qc_path / 'losses_over_time.png',
                bbox_inches="tight", pad_inches=1.)

    incumbent_trajectory = res.get_incumbent_trajectory()

    config = id2config[incumbent]['config']
    print('Best found configuration:', config)

    with open((modelpath / (run_id + '-encoder_config.json')), "w") as handle:
        json.dump(config, handle)


    # Run model once on ideal and save
    x_valid = pd.read_csv(args.valid_data, index_col=0).astype(
        'float32')

    x_test = {}
    if args.test_data:
        for t in args.test_data:
            x_test[t] = pd.read_csv(Path(t), index_col=0).astype('float32')


    feature_order = x_train.var(axis=0).sort_values(ascending=False).index

    train = x_train.loc[:, feature_order[0:config['nfeatures']]]
    valid = x_valid.loc[:, feature_order[0:config['nfeatures']]]

    test = {}
    if args.test_data:
        for t in args.test_data:
            test[t] = x_test[t].loc[:, feature_order[0:config['nfeatures']]]

    model = ec.build_autoencoder(train, valid, config)

    model.compile(loss=tf.keras.losses.mean_squared_error,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=config['lr']))
    
    if args.scheduler == '1cycle':
        onecycle_cb = ec.OneCycleLearningRate(np.ceil(train.shape[0] / config['batchsize']) * args.max_budget,
                                          max_rate=config['lr'])
        callbacks = [onecycle_cb]
        
    elif args.scheduler == '1cycle2':
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(ec.onecycle_lr(config['lr'], args.max_budget, 10, model=model))
        momentum_scheduler = ec.OneCycleMomentumCallback(args.max_budget, 10)
        callbacks = [lr_scheduler, momentum_scheduler]
        
    elif args.scheduler == 'exponential':
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(ec.exponential_decay(lr0=config['lr'], s=config['s']))
        callbacks = [lr_scheduler]
        
    elif args.scheduler == 'power':
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        ec.decayed_learning_rate(lr0=config['lr'], s=config['s']))
        callbacks = [lr_scheduler]


    history = model.fit(train, train,
                        batch_size=config['batchsize'],
                        epochs=int(args.max_budget),
                        verbose=0,
                        callbacks=callbacks,
                        validation_data=(valid, valid))

    model.save(qc_path / 'model')

    # Get latent encoder
    encoder = tf.keras.Model(inputs=[model.layers[0].input], outputs=[model.layers[0].output])

    with open((modelpath / (run_id + '-encoder_model_architecture.json')), "w") as handle:
        json.dump(encoder.to_json(), handle)


    # serialize weights
    list_weights = [w.tolist() for w in encoder.get_weights()]
    with open((modelpath / (run_id + '-encoder_model_weights.json')), "w") as handle:
        json.dump(list_weights, handle)
    # encoder.save(modelpath / (run_id + '-model'))

    # Get latent data representation
    val = encoder.predict(valid)
    valid.to_csv((modelpath / 'valid_trimmed.csv'))
    trai = encoder.predict(train)
    tes = {}
    if args.test_data:
        for t in args.test_data:
            tes[t] = encoder.predict(test[t])

    # Save latent data representation
    pd.DataFrame(trai, index=train.index).to_csv(dataset_path / (run_id + '-train.csv'))
    pd.DataFrame(val, index=valid.index).to_csv(dataset_path / (run_id + '-valid.csv'))
    if args.test_data:
        for t in args.test_data:
            n = re.match(r"(?P<prep>preprocess_\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2}_)(?P<root>.*)(?P<tail>-\w{2,5}_\w+.csv)",
                         Path(t).name)
            if n:
                pd.DataFrame(tes[t], index=test[t].index).to_csv(dataset_path / (
                        run_id + '-' + n.group('root') + '.csv'))
            else:
                pd.DataFrame(tes[t], index=test[t].index).to_csv(dataset_path / (
                        run_id + '-' + Path(t).stem + '.csv'))


    # mean loss since model is sum of individual
    train_score = model.evaluate(train, train, verbose=0) 
    valid_score = model.evaluate(valid, valid, verbose=0)

    if args.modeltype == 'autoencoder':
        # determine latent space for valid dataset to calculate metrics
        latent = tf.keras.Model(inputs=[model.layers[0].input], outputs=[model.layers[0].output])

    metricsdict = {
        'valid loss': valid_score,
        'train loss': train_score,
        'features': feature_order[0:config['nfeatures']].to_list()
    }

    with open((modelpath / (run_id + '-encoder_metrics.json')), "w") as handle:
        json.dump(metricsdict, handle)

    argsdict = {'train_data': args.train_data,
                'valid_data': args.valid_data,
                'output_directory': args.output_directory.as_posix(),
                'min_budget': args.min_budget,
                'max_budget': args.max_budget,
                'n_iterations': args.n_iterations,
                'debug': args.debug,
                'nic_name': args.nic_name,
                'scheduler': args.scheduler,
                'modeltype': args.modeltype}

    with open((modelpath / (run_id + '-encoder_input_args.json')), "w") as handle:
        json.dump(argsdict, handle)

print("\033[1;32m Autoencoder Run: ", run_id)

