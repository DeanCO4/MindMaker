#(c) Copywrite 2020 Aaron Krumins

from flask import Flask
import logging.handlers
import socketio
import json
import numpy as np
import sys
import os
from gym import spaces
from random import randint
import ast
import math
import gym
import torch as th
#import tensorboard
#import tensorflow as tf
#from torch.utils.tensorboard import SummaryWriter
import stable_baselines3
from stable_baselines3.dqn.policies import MlpPolicy as DqnMlpPolicy
from stable_baselines3.sac.policies import MlpPolicy as SacMlpPolicy
from stable_baselines3.td3.policies import MlpPolicy as Td3MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3 import DQN, A2C, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env

from tensorboard import version; print(version.VERSION)
# set async_mode to 'threading', 'eventlet', 'gevent' or 'gevent_uwsgi' to
# force a mode else, the best mode is selected automatically from what's
# installed
async_mode = None

# Set up a specific logger with our desired output level
# logging.disable(sys.maxsize)

sio = socketio.Server(logger=True)
app = Flask(__name__)
app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)

thread = None

observations = "NaN"
UEreward = "0"
UEdone = False
maxactions = 0
reward = 0
obsflag = 0
inf = math.inf
actionspace = "Nan"
observationspace = "Nan"
results = os.getenv('APPDATA')
conactionspace = ''
disactionspace = ''


# if getattr(sys, 'frozen', False):
# application_path = os.path.dirname(sys.executable)
# elif __file__:
# application_path = os.path.dirname(__file__)
# directory = "\\MindMaker"


def check_obs(self):
    global obsflag
    if obsflag == 1:
        # observations recieved from UE, continue training
        obsflag = 0
    else:
        # observations not recieved yet, check again in a half second
        sio.sleep(.06)
        check_obs(self)


class UnrealEnvWrap(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is an env wrapper that recieves any environmental variables from UE and shapes into a format for OpenAI Gym
    """
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {'render.modes': ['console']}

    def __init__(self, ):
        super(UnrealEnvWrap, self).__init__()
        global maxactions
        global conactionspace
        global disactionspace
        # global minaction
        # global maxaction
        global actionspace
        global observationspace
        # print (minaction)
        # print (maxaction)
        print(conactionspace)

        print(actionspace)
        if conactionspace:
            print("continous action space")
            actionspace = "spaces.Box(" + actionspace + ")"
            observationspace = "spaces.Box(" + observationspace + ")"
            # self.action_space = spaces.Box(ast.literal_eval(actionspace))
            self.action_space = eval(actionspace)
            # self.agent_pos = randint(0, maxactions)
            self.agent_pos = randint(0, 100)
            # low = np.array([-2,0,-100])
            # high = np.array([2, 100, 100])
            self.observation_space = eval(observationspace)
        elif disactionspace:
            # Initialize the agent with a random action
            print("discrete action space")
            actionspace = int(actionspace)
            self.agent_pos = randint(0, actionspace)

            # Define action and observation space
            # They must be gym.spaces objects
            # Example when using discrete actions, we have two: left and right
            n_actions = actionspace
            self.action_space = spaces.Discrete(n_actions)
            observationspace = "spaces.Box(" + observationspace + ")"
            # The observation will be all environment variables from UE that agent is tracking
            # n_actionsforarray = n_actions - 1
            # low = np.array([0,0])
            # high = np.array([n_actionsforarray, n_actionsforarray])
            # self.observation_space = spaces.Box(low, high, dtype=np.float32)
            self.observation_space = eval(observationspace)
        else:
            logmessages = "No action space type selected"
            sio.emit('messages', logmessages)

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        # Initialize the agent with a random action
        self.observation_space = [0]
        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return np.array([self.observation_space])

    # sending actions to UE and recieving observations in response to those actions
    def step(self, action):
        global observations
        global reward
        global UEreward
        global UEdone
        global obsflag
        obsflag = 0

        # send actions to UE as they are chosen by the RL algorityhm
        straction = str(action)
        print("action:", straction)
        sio.emit('recaction', straction)
        # After sending action, we enter a pause loop until we recieve a response from UE with the observations
        #After sending action, we enter a pause loop until we recieve a response from UE with the observations
        for i in range(10000):
            if obsflag == 1:
                obsflag = 0
                break
            else:
                sio.sleep(.06)

        # load the observations recieved from UE4
        arrayobs = ast.literal_eval(observations)
        self.observation_space = arrayobs
        print(arrayobs)
        done = bool(UEdone)
        reward = float(UEreward)
        print("reward", reward)
        print(UEdone)
        if done:
            print("Im rrestarting now how fun")
            reward = 0
        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return np.array([self.observation_space]).astype(np.float32), reward, done, info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

    def close(self):
        exit(1)


@sio.event
def disconnect_request(sid):
    sio.disconnect(sid)
    exit()


@sio.event
def connect(*_):
    print("Connected To Unreal Engine")


@sio.event
def disconnect(_):
    print('Disconnected From Unreal Engine, Exiting MindMaker')


@sio.on('launchmindmaker')
def recieve(sid, data):
    global UEdone
    global reward
    global maxactions
    global conactionspace
    global disactionspace
    # global minaction
    # global maxaction
    global actionspace
    global observationspace
    json_input = json.loads(data)
    actionspace = json_input['actionspace']
    observationspace = json_input['observationspace']
    # minaction = json_input['minaction']
    # maxaction = json_input['maxaction']
    # maxactions = json_input['maxactions']
    trainepisodes = json_input['trainepisodes']
    evalepisodes = json_input['evalepisodes']
    loadmodel = json_input['loadmodel']
    savemodel = json_input['savemodel']
    modelname = json_input['modelname']
    algselected = json_input['algselected']
    usecustomparams = json_input['customparams']
    a2cparams = json_input['a2cparams']
    dqnparams = json_input['dqnparams']
    ddpgparams = json_input.get('ddpgparams', {})
    ppoparams = json_input.get('ppoparams', {})
    sacparams = json_input['sacparams']
    td3params = json_input['td3params']
    conactionspace = json_input['conactionspace']
    disactionspace = json_input['disactionspace']
    loadforprediction = json_input['loadforprediction']
    loadmodelname = json_input['loadmodelname']
    UEdone = json_input['done']
    env = UnrealEnvWrap()
    # wrap it
    env = make_vec_env(lambda: env, n_envs=1)
    print("save model value:", savemodel)
    print("load model value:", loadmodel)
    print("Total Training Episodes:", trainepisodes)

    path = results + "\\" + modelname
    loadpath = results + "\\" + loadmodelname

    if loadmodel == 'true':
        # Load the trained agent
        if algselected == 'DQN':
            model = DQN.load(loadpath, env=env)
        elif algselected == 'A2C':
            model = A2C.load(loadpath, env=env)
        elif algselected == 'DDPG':
            from stable_baselines3.ddpg.policies import MlpPolicy as DdpgMlpPolicy
            from stable_baselines3 import DDPG
            #print("DDPG requires Microsoft Open MPI be installed on your system")
            model = DDPG.load(loadpath, env=env)
        elif algselected == 'PPO':
            from stable_baselines3 import PPO
            model = PPO.load(loadpath, env=env)
            #print("lOADDED PPO model")
        elif algselected == 'SAC':
            model = SAC.load(loadpath, env=env)
        elif algselected == 'TD3':
            model = TD3.load(loadpath, env=env)
        else:
            model = {}

        print("Loading the Agent for Continous Training")
        logmessages = "Loading the Agent for Continous Training"
        sio.emit('messages', logmessages)
        #_ = env.reset()
        # intaction = 0
        # Begin strategic behvaior
        #model.set_env(env)
        model.learn(total_timesteps=trainepisodes)
        if savemodel == 'true':
            # Save the agent
            model.save(path)
            print("Saving the Trained Agent")
            logmessages = "The trained model was saved"
            sio.emit('messages', logmessages)
        quit()
        
    if loadforprediction == 'true':

        # Load the trained agent
        if algselected == 'DQN':
            model = DQN.load(loadpath)
        elif algselected == 'A2C':
            model = A2C.load(loadpath)
        elif algselected == 'DDPG':
            from stable_baselines3.ddpg.policies import MlpPolicy as DdpgMlpPolicy
            from stable_baselines3 import DDPG
            #print("DDPG requires Microsoft Open MPI be installed on your system")
            model = DDPG.load(loadpath)
        elif algselected == 'PPO':
            from stable_baselines3 import PPO
            model = PPO.load(loadpath)
        elif algselected == 'SAC':
            model = SAC.load(loadpath)
        elif algselected == 'TD3':
            model = TD3.load(loadpath)
        else:
            model = {}

        print("Loading the Agent for Prediction")
        logmessages = "Loading the Agent for Prediction"
        sio.emit('messages', logmessages)
        #_ = env.reset()
        env.render(mode='console')
        # env.render()

        obs = env.reset()
        #model.set_env(env)
        evalcomplete = evalepisodes + 2
        print(evalcomplete)
        for step in range(evalcomplete):
            action, _ = model.predict(obs, deterministic=True)
            intaction = action[0]
            print("Action: ", intaction)
            obs, reward, done, info = env.step(action)
            print('obs=', obs, 'reward=', reward, 'done=', done)
            if step == evalepisodes:
                print(step)
                logmessages = "Evaluation Complete"
                sio.emit('messages', logmessages)
        quit()
    #        for i in range(evalepisodes):
    #            action, _ = model.predict(obs, deterministic=True)

        # intaction = 0
        # Begin strategic behvaior
        
        #model.learn(total_timesteps=trainepisodes)
        
        
 

    else:
        # Train the agent with different algorityhms from stable baselines

        # model = DQN(MlpPolicy, env, verbose=1, tensorboard_log="./DQN_newobservations/")
        print("alg selected:", algselected)
        print("use custom:", usecustomparams)

        if (algselected == 'DQN') and (usecustomparams == 'true'):
            learning_rate = dqnparams["learning rate"]
            buffer_size = dqnparams["buffer size"]
            learning_starts = dqnparams["learning starts"]
            batch_size = dqnparams["batch size"]
            tau = dqnparams["tau"]
            gamma = dqnparams["gamma"]
            train_freq = dqnparams["train freq"]
            gradient_steps = dqnparams["gradient steps"]
            replay_buffer_class = dqnparams["replay buffer class"]
            replay_buffer_kwargs  = dqnparams["replay buffer kwargs"]
            optimize_memory_usage = dqnparams["optimize memory usage"]
            target_update_interval = dqnparams["target update interval"]
            exploration_fraction = dqnparams["exploration fraction"]
            exploration_initial_eps = dqnparams["exploration initial eps"]
            exploration_final_eps = dqnparams["exploration final eps"]
            max_grad_norm = dqnparams["max grad norm"]
            tensorboard_log = dqnparams["tensorboard log"]
            create_eval_env = dqnparams["create eval env"]
            verbose = dqnparams["verbose"]
            seed = dqnparams["seed"]
            device = dqnparams["device"]
            _init_setup_model = dqnparams["init setup model"]
            policy = dqnparams["policy"]
            #policy_kwargs = dqnparams["policy kwargs"]
            
            act_func = dqnparams["activation func"]
            network_arch = dqnparams["network arch"]
            

            #CAN ONLY PASS IN SINGLE ARGUEMENT VALUES FOR NETWORK ARCH, NO DICS
            if act_func == 'th.nn.ReLU':
                 policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch = ast.literal_eval(network_arch))

            elif act_func == 'th.nn.LeakyReLU':
                 policy_kwargs = dict(activation_fn=th.nn.LeakyReLU, net_arch = ast.literal_eval(network_arch))
            elif act_func == 'th.nn.Tanh':
                 policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch = ast.literal_eval(network_arch))
            elif act_func == 'th.nn.ReLU6':
                 policy_kwargs = dict(activation_fn=th.nn.ReLU6, net_arch = ast.literal_eval(network_arch))
            elif act_func == 'th.nn.SELU':
                 policy_kwargs = dict(activation_fn=th.nn.SELU, net_arch = ast.literal_eval(network_arch))
            elif act_func == 'th.nn.ELU':
                 policy_kwargs = dict(activation_fn=th.nn.ELU, net_arch = ast.literal_eval(network_arch))
            else:
                 policy_kwargs = dict(net_arch = ast.literal_eval(network_arch))
                 
            model = DQN(policy, env, gamma=gamma, learning_rate=learning_rate, verbose=verbose,
                        tensorboard_log=tensorboard_log, _init_setup_model=_init_setup_model,
                        seed=seed,
                        buffer_size=buffer_size, exploration_fraction=exploration_fraction,
                        exploration_final_eps=exploration_final_eps,
                        exploration_initial_eps=exploration_initial_eps, batch_size=batch_size, tau = tau,
                        train_freq=train_freq, gradient_steps = gradient_steps, replay_buffer_class = ast.literal_eval(replay_buffer_class), replay_buffer_kwargs = ast.literal_eval(replay_buffer_kwargs), optimize_memory_usage = optimize_memory_usage, target_update_interval = target_update_interval, max_grad_norm = max_grad_norm, create_eval_env  = create_eval_env, device = device, learning_starts=learning_starts, policy_kwargs = policy_kwargs)
            
            # model = DQN(policy, env, learning_rate=0.0001, buffer_size=1000000, learning_starts=50000, batch_size=32, tau=1.0, gamma=0.99, train_freq=4, gradient_steps=1, replay_buffer_class=None, replay_buffer_kwargs=None, optimize_memory_usage=False, target_update_interval=10000, exploration_fraction=0.1, exploration_initial_eps=1.0, exploration_final_eps=0.05, max_grad_norm=10, tensorboard_log=None, create_eval_env=False, policy_kwargs=None, verbose=0, seed=None, device='auto', _init_setup_model=True)
            
            print("Custom DQN training in process...")
            model.learn(total_timesteps=trainepisodes)
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")

        elif algselected == 'DQN':
            model = DQN('MlpPolicy', env, verbose=1)
            print("DQN training in process...")
            model.learn(total_timesteps=trainepisodes)
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
        elif (algselected == 'A2C') and (usecustomparams == 'true'):
            print(a2cparams)
            policy = a2cparams["policy"]
            learning_rate = a2cparams["learning rate"]
            n_steps = a2cparams["n steps"]
            gamma = a2cparams["gamma"]
            gae_lambda = a2cparams["gae lambda"]
            ent_coef = a2cparams["ent coef"]
            vf_coef = a2cparams["vf coef"]
            max_grad_norm = a2cparams['max grad norm']
            rms_prop_eps = a2cparams['rms prop eps']
            use_rms_prop = a2cparams['use rms prop']
            use_sde = a2cparams['use sde']
            sde_sample_freq = a2cparams['sde sample freq']
            normalize_advantage = a2cparams['normalize advantage']
            tensorboard_log = a2cparams["tensorboard log"]
            create_eval_env = a2cparams["create eval env"]
            #policy_kwargs = a2cparams["policy kwargs"]
            verbose = a2cparams["verbose"]
            seed = a2cparams["seed"]
            device = a2cparams["device"]
            _init_setup_model = a2cparams["init setup model"]
            act_func = a2cparams["activation func"]
            network_arch = a2cparams["network arch"]
            

            #CAN ONLY PASS IN SINGLE ARGUEMENT VALUES FOR NETWORK ARCH, NO DICS
            if act_func == 'th.nn.ReLU':
                 policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch = ast.literal_eval(network_arch))

            elif act_func == 'th.nn.LeakyReLU':
                 policy_kwargs = dict(activation_fn=th.nn.LeakyReLU, net_arch = ast.literal_eval(network_arch))
            elif act_func == 'th.nn.Tanh':
                 policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch = ast.literal_eval(network_arch))
            elif act_func == 'th.nn.ReLU6':
                 policy_kwargs = dict(activation_fn=th.nn.ReLU6, net_arch = ast.literal_eval(network_arch))
            elif act_func == 'th.nn.SELU':
                 policy_kwargs = dict(activation_fn=th.nn.SELU, net_arch = ast.literal_eval(network_arch))
            elif act_func == 'th.nn.ELU':
                 policy_kwargs = dict(activation_fn=th.nn.ELU, net_arch = ast.literal_eval(network_arch))
            else:
                 policy_kwargs = dict(net_arch = ast.literal_eval(network_arch))
            
            #act_func = a2cparams["act func"]
            #alpha = a2cparams["alpha"]
            #epsilon = a2cparams["epsilon"]
            #lr_schedule = a2cparams["lr schedule"]
            #full_tensorboard_log = a2cparams["full tensorboard log"]
            #n_cpu_tf_sess = a2cparams["n cpu tf sess"]
            
            
            #network_arch = a2cparams["network arch"]
            
            #print('hello, world!')

            model = A2C(policy = policy, env = env, learning_rate = learning_rate, n_steps = n_steps, gamma = gamma, gae_lambda = gae_lambda, ent_coef = ent_coef, vf_coef = vf_coef, max_grad_norm = max_grad_norm, rms_prop_eps= rms_prop_eps, use_rms_prop= use_rms_prop, use_sde = use_sde , sde_sample_freq= sde_sample_freq, normalize_advantage =normalize_advantage, tensorboard_log= tensorboard_log, create_eval_env= create_eval_env, policy_kwargs = policy_kwargs, verbose = verbose, seed= seed, device= device, _init_setup_model= _init_setup_model)

            print("Custom A2C training in process...")
            model.learn(total_timesteps=trainepisodes)
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
        elif algselected == 'A2C':
            model = A2C('MlpPolicy', env, verbose=1)
            print("A2C training in process...")
            model.learn(total_timesteps=trainepisodes)
            
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")


        elif (algselected == 'DDPG') and (usecustomparams == 'true'):
            from stable_baselines3.ddpg.policies import MlpPolicy as DdpgMlpPolicy
            from stable_baselines3 import DDPG
            #print("DDPG requires Microsoft Open MPI be installed on your system")
            policy = ddpgparams["policy"]
            learning_rate = ddpgparams["learning rate"]
            buffer_size = ddpgparams["buffer size"]
            learning_starts = ddpgparams["learning starts"]
            batch_size = ddpgparams["batch size"]
            tau = ddpgparams["tau"]
            gamma = ddpgparams["gamma"]
            train_freq = ddpgparams["train freq"]
            gradient_steps = ddpgparams["gradient steps"]
            action_noise = ddpgparams["action noise"]
            replay_buffer_class = ddpgparams["replay buffer class"]
            replay_buffer_kwargs = ddpgparams["replay buffer kwargs"]
            optimize_memory_usage = ddpgparams["optimize memory usage"]
            create_eval_env = ddpgparams["create eval env"]
            verbose = ddpgparams["verbose"]
            seed = ddpgparams["seed"]
            device = ddpgparams["device"]
            _init_setup_model = ddpgparams["init setup model"]
            tensorboard_log = ddpgparams["tensorboard log"]
            #policy_kwargs = ddpgparams["policy kwargs"]
            
            act_func = ddpgparams["activation func"]
            network_arch = ddpgparams["network arch"]
            

            #CAN ONLY PASS IN SINGLE ARGUEMENT VALUES FOR NETWORK ARCH, NO DICS
            if act_func == 'th.nn.ReLU':
                 policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch = ast.literal_eval(network_arch))

            elif act_func == 'th.nn.LeakyReLU':
                 policy_kwargs = dict(activation_fn=th.nn.LeakyReLU, net_arch = ast.literal_eval(network_arch))
            elif act_func == 'th.nn.Tanh':
                 policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch = ast.literal_eval(network_arch))
            elif act_func == 'th.nn.ReLU6':
                 policy_kwargs = dict(activation_fn=th.nn.ReLU6, net_arch = ast.literal_eval(network_arch))
            elif act_func == 'th.nn.SELU':
                 policy_kwargs = dict(activation_fn=th.nn.SELU, net_arch = ast.literal_eval(network_arch))
            elif act_func == 'th.nn.ELU':
                 policy_kwargs = dict(activation_fn=th.nn.ELU, net_arch = ast.literal_eval(network_arch))
            else:
                 policy_kwargs = dict(net_arch = ast.literal_eval(network_arch))
            
            
            
            model = DDPG(policy, env, learning_rate = learning_rate, buffer_size = buffer_size, learning_starts = learning_starts, batch_size = batch_size, tau = tau, gamma= gamma, train_freq = ast.literal_eval(train_freq), gradient_steps = gradient_steps, action_noise = ast.literal_eval(action_noise), replay_buffer_class = ast.literal_eval(replay_buffer_class), replay_buffer_kwargs = ast.literal_eval(replay_buffer_kwargs), optimize_memory_usage = optimize_memory_usage, tensorboard_log = tensorboard_log, create_eval_env = create_eval_env, verbose = verbose, seed = ast.literal_eval(seed), device = device, _init_setup_model = _init_setup_model, policy_kwargs = policy_kwargs )
            
            
            
            
            # #model = DDPG(eval(policyval), env, gamma=gammaval, verbose=verboseval, tensorboard_log=tensorboard_logval,
                         # _init_setup_model=_init_setup_modelval, policy_kwargs=policy_kwargsval,
                         # seed=ast.literal_eval(seedval),
                         # buffer_size=ast.literal_eval(buffer_sizeval),
                         # tau=tauval,
                         # action_noise=ast.literal_eval(action_noiseval))

            print("Custom DDPG training in process...")
            model.learn(total_timesteps=trainepisodes)
            print("DDPG training complete")
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
        elif algselected == 'DDPG':
            from stable_baselines3.ddpg.policies import MlpPolicy as DdpgMlpPolicy
            from stable_baselines3 import DDPG
            #print("DDPG requires Microsoft Open MPI be installed on your system")
            # the noise objects for DDPG
            n_actions = env.action_space.shape[-1]
            # param_noise = None
            action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
            model = DDPG(DdpgMlpPolicy, env, verbose=1, action_noise=action_noise)
            print("DDPG training in process...")
            model.learn(total_timesteps=trainepisodes)
            print("DDPG training complete")
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")

        elif (algselected == 'PPO') and (usecustomparams == 'true'):
            from stable_baselines3 import PPO
            policy = ppoparams["policy"]
            learning_rate = ppoparams["learning rate"]
            n_steps = ppoparams["n steps"]
            batch_size = ppoparams["batch size"]
            n_epochs = ppoparams["n epochs"]
            gamma = ppoparams["gamma"]
            gae_lambda = ppoparams["gae lambda"]
            clip_range = ppoparams["clip range"]
            clip_range_vf = ppoparams["clip range vf"]
            normalize_advantage = ppoparams["normalize advantage"]
            ent_coef = ppoparams["ent coef"]
            vf_coef = ppoparams["vf coef"]
            max_grad_norm = ppoparams["max grad norm"]
            use_sde = ppoparams["use sde"]
            sde_sample_freq = ppoparams["sde sample freq"]
            target_kl = ppoparams["target kl"]
            tensorboard_log = ppoparams["tensorboard log"]
            create_eval_env = ppoparams["create eval env"]
            verbose = ppoparams["verbose"]
            seed = ppoparams["seed"]
            device = ppoparams["device"]
            _init_setup_model = ppoparams["init setup model"]
            act_func = ppoparams["activation func"]
            network_arch = ppoparams["network arch"]
            #policy_kwargs = ppoparams["policy kwargs"]

            #CAN ONLY PASS IN SINGLE ARGUEMENT VALUES FOR NETWORK ARCH, NO DICS
            if act_func == 'th.nn.ReLU':
                 policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch = ast.literal_eval(network_arch))

            elif act_func == 'th.nn.LeakyReLU':
                 policy_kwargs = dict(activation_fn=th.nn.LeakyReLU, net_arch = ast.literal_eval(network_arch))
            elif act_func == 'th.nn.Tanh':
                 policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch = ast.literal_eval(network_arch))
            elif act_func == 'th.nn.ReLU6':
                 policy_kwargs = dict(activation_fn=th.nn.ReLU6, net_arch = ast.literal_eval(network_arch))
            elif act_func == 'th.nn.SELU':
                 policy_kwargs = dict(activation_fn=th.nn.SELU, net_arch = ast.literal_eval(network_arch))
            elif act_func == 'th.nn.ELU':
                 policy_kwargs = dict(activation_fn=th.nn.ELU, net_arch = ast.literal_eval(network_arch))
            else:
                 policy_kwargs = dict(net_arch = ast.literal_eval(network_arch))

         
            model = PPO(policy, env, learning_rate = learning_rate, n_steps = n_steps, batch_size = batch_size, gamma = gamma, gae_lambda = gae_lambda,  clip_range = clip_range, clip_range_vf = ast.literal_eval(clip_range_vf), normalize_advantage = normalize_advantage, ent_coef = ent_coef, vf_coef = vf_coef, max_grad_norm = max_grad_norm, use_sde = use_sde, sde_sample_freq = sde_sample_freq, tensorboard_log = tensorboard_log, create_eval_env = create_eval_env, policy_kwargs = policy_kwargs, verbose = verbose, seed = seed, device = device, _init_setup_model = _init_setup_model  )

            print("Custom PPO training in process...")
            model.learn(total_timesteps=trainepisodes)
            print("PPO training complete")
            if savemodel == 'true':
                # Save the agent
                path = results + "\\" + modelname
                model.save(path)
                print("Saving the Trained Agent")
                logmessages = "The trained model was saved"
                sio.emit('messages', logmessages)

        elif algselected == 'PPO':
            from stable_baselines3 import PPO
            model = PPO('MlpPolicy', env, verbose=1)
            print("PPO training in process...")
            model.learn(total_timesteps=trainepisodes)
            print("PPO training complete")
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
                logmessages = "The trained model was saved"
                sio.emit('messages', logmessages)

        elif (algselected == 'SAC') and (usecustomparams == 'true'):
            policy = sacparams["policy"]
            learning_rate = sacparams["learning rate"]
            buffer_size = sacparams["buffer size"]
            learning_starts = sacparams["learning starts"]
            batch_size = sacparams["batch size"]
            tau = sacparams["tau"]
            gamma = sacparams["gamma"]
            train_freq = sacparams["train freq"]
            gradient_steps = sacparams["gradient steps"]
            action_noise = sacparams["action noise"]
            replay_buffer_class = sacparams["replay buffer class"]
            replay_buffer_kwargs = sacparams["replay buffer kwargs"]
            optimize_memory_usage = sacparams["optimize memory usage"]
            ent_coef = sacparams["ent coef"]
            target_update_interval = sacparams["target update interval"]
            target_entropy = sacparams["target entropy"]
            use_sde = sacparams["use sde"]
            sde_sample_freq = sacparams["sde sample freq"]
            use_sde_at_warmup = sacparams["use sde at warmup"]
            create_eval_env = sacparams["create eval env"]
            #policy_kwargs = sacparams["policy kwargs"]
            verbose = sacparams["verbose"]
            seed = sacparams["seed"]
            device = sacparams["device"]
            _init_setup_model = sacparams["init setup model"]
            tensorboard_log = sacparams["tensorboard log"]
            
            act_func = sacparams["activation func"]
            network_arch = sacparams["network arch"]
            

            #CAN ONLY PASS IN SINGLE ARGUEMENT VALUES FOR NETWORK ARCH, NO DICS
            if act_func == 'th.nn.ReLU':
                 policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch = ast.literal_eval(network_arch))

            elif act_func == 'th.nn.LeakyReLU':
                 policy_kwargs = dict(activation_fn=th.nn.LeakyReLU, net_arch = ast.literal_eval(network_arch))
            elif act_func == 'th.nn.Tanh':
                 policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch = ast.literal_eval(network_arch))
            elif act_func == 'th.nn.ReLU6':
                 policy_kwargs = dict(activation_fn=th.nn.ReLU6, net_arch = ast.literal_eval(network_arch))
            elif act_func == 'th.nn.SELU':
                 policy_kwargs = dict(activation_fn=th.nn.SELU, net_arch = ast.literal_eval(network_arch))
            elif act_func == 'th.nn.ELU':
                 policy_kwargs = dict(activation_fn=th.nn.ELU, net_arch = ast.literal_eval(network_arch))
            else:
                 policy_kwargs = dict(net_arch = ast.literal_eval(network_arch))

            model = SAC(policy, env, gamma=gamma, learning_rate=learning_rate, buffer_size=buffer_size, learning_starts=learning_starts,
                        train_freq=train_freq, batch_size=batch_size, tau=tau, ent_coef=ent_coef, target_update_interval=target_update_interval,
                        gradient_steps=gradient_steps, target_entropy=target_entropy, action_noise=ast.literal_eval(action_noise), verbose=verbose,
                        tensorboard_log=tensorboard_log, _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs, seed=ast.literal_eval(seed), replay_buffer_class =ast.literal_eval(replay_buffer_class), replay_buffer_kwargs =ast.literal_eval(replay_buffer_kwargs), optimize_memory_usage = optimize_memory_usage, use_sde = use_sde, sde_sample_freq = sde_sample_freq, use_sde_at_warmup = use_sde_at_warmup, create_eval_env = create_eval_env, device = device )

            print("Custom SAC training in process...")
            model.learn(total_timesteps=trainepisodes)
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
                logmessages = "The trained model was saved"
                sio.emit('messages', logmessages)

        elif algselected == 'SAC':
            model = SAC(SacMlpPolicy, env, verbose=1)
            print("SAC training in process...")
            model.learn(total_timesteps=trainepisodes, log_interval=10)
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
                logmessages = "The trained model was saved"
                sio.emit('messages', logmessages)

        elif (algselected == 'TD3') and (usecustomparams == 'true'):
            policy = td3params["policy"]
            learning_rate = td3params["learning rate"]
            buffer_size = td3params["buffer size"]
            learning_starts = td3params["learning starts"]
            batch_size = td3params["batch size"]
            tau = td3params["tau"]
            gamma = td3params["gamma"]
            train_freq = td3params["train freq"]
            gradient_steps = td3params["gradient steps"]
            action_noise = td3params["action noise"]
            replay_buffer_class = td3params["replay buffer class"]
            replay_buffer_kwargs = td3params["replay buffer kwargs"]
            optimize_memory_usage = td3params["optimize memory usage"]
            policy_delay = td3params["policy delay"]
            target_policy_noise = td3params["target policy noise"]
            target_noise_clip = td3params["target noise clip"]
            create_eval_env = td3params["create eval env"]
            #policy_kwargs = td3params["policy kwargs"]
            verbose = td3params["verbose"]
            seed = td3params["seed"]
            device = td3params["device"]
            _init_setup_model = td3params["init setup model"]
            tensorboard_log = td3params["tensorboard log"]
            act_func = td3params["activation func"]
            network_arch = td3params["network arch"]
            

            #CAN ONLY PASS IN SINGLE ARGUEMENT VALUES FOR NETWORK ARCH, NO DICS
            if act_func == 'th.nn.ReLU':
                 policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch = ast.literal_eval(network_arch))

            elif act_func == 'th.nn.LeakyReLU':
                 policy_kwargs = dict(activation_fn=th.nn.LeakyReLU, net_arch = ast.literal_eval(network_arch))
            elif act_func == 'th.nn.Tanh':
                 policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch = ast.literal_eval(network_arch))
            elif act_func == 'th.nn.ReLU6':
                 policy_kwargs = dict(activation_fn=th.nn.ReLU6, net_arch = ast.literal_eval(network_arch))
            elif act_func == 'th.nn.SELU':
                 policy_kwargs = dict(activation_fn=th.nn.SELU, net_arch = ast.literal_eval(network_arch))
            elif act_func == 'th.nn.ELU':
                 policy_kwargs = dict(activation_fn=th.nn.ELU, net_arch = ast.literal_eval(network_arch))
            else:
                 policy_kwargs = dict(net_arch = ast.literal_eval(network_arch))

            model = TD3(policy, env, gamma=gamma, learning_rate=learning_rate, verbose=verbose,
                        tensorboard_log=tensorboard_log, _init_setup_model=_init_setup_model,
                        policy_kwargs=policy_kwargs, seed=ast.literal_eval(seed), buffer_size=buffer_size,
                        tau=tau, target_noise_clip=target_noise_clip,
                        policy_delay=policy_delay, batch_size=batch_size, train_freq=train_freq,
                        gradient_steps=gradient_steps, learning_starts=learning_starts,
                        action_noise=ast.literal_eval(action_noise), target_policy_noise=target_policy_noise, replay_buffer_class = ast.literal_eval(replay_buffer_class), replay_buffer_kwargs = ast.literal_eval(replay_buffer_kwargs), optimize_memory_usage = optimize_memory_usage, create_eval_env = create_eval_env, device = device )

            print("Custom TD3 training in process...")
            model.learn(total_timesteps=trainepisodes)
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
                logmessages = "The trained model was saved"
                sio.emit('messages', logmessages)

        elif algselected == 'TD3':
            # The noise objects for TD3
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
            model = TD3(Td3MlpPolicy, env, action_noise=action_noise, verbose=1)
            print("TD3 training in process...")
            model.learn(total_timesteps=trainepisodes)
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
                logmessages = "The trained model was saved"
                sio.emit('messages', logmessages)
        else:
            model = {}
            print("No learning algorithm selected for training with")
            logmessages = "No learning algorithm selected for training with"
            sio.emit('messages', logmessages)
            sio.disconnect(sid)

        # Test the trained agent, (currently not needed, all testing occurs in Unreal itself)

        #env.render(mode='console')
        # env.render()

        #obs = env.reset()
        print("Training complete")
        logmessages = "Training complete"
        sio.emit('messages', logmessages)
        # intaction = 0
        # Begin strategic behvaior
        evalcomplete = evalepisodes + 2
        print(evalcomplete)
        for step in range(evalcomplete):
            action, _ = model.predict(obs, deterministic=True)
            intaction = action[0]
            print("Action: ", intaction)
            obs, reward, done, info = env.step(action)
            print('obs=', obs, 'reward=', reward, 'done=', done)
            if step == evalepisodes:
                print(step)
                logmessages = "Evaluation Complete"
                sio.emit('messages', logmessages)

    sio.disconnect(sid)


# recieves observations and reward from Unreal Engine
@sio.on('sendobs')
def sendobs(_, obsdata):
    global obsflag
    global observations
    global UEreward
    global UEdone

    obsflag = 1
    json_input = json.loads(obsdata)
    observations = json_input['observations']
    UEreward = json_input['reward']
    UEdone = json_input['done']


# This sets up the server connection, with UE acting as the client in a socketIO relationship, will default to eventlet
if __name__ == '__main__':
    if sio.async_mode == 'threading':
        # deploy with Werkzeug
        print("1 ran")
        app.run(threaded=True)

    elif sio.async_mode == 'eventlet':
        # deploy with eventlet
        import eventlet
        import eventlet.wsgi

        logging.disable(sys.maxsize)
        print("MindMaker running, waiting for Unreal Engine to connect")
        eventlet.wsgi.server(eventlet.listen(('', 5001)), app)
    elif sio.async_mode == 'gevent':
        # deploy with gevent
        from gevent import pywsgi

        # from geventwebsocket.handler import WebSocketHandler
        try:
            websocket = True
            print("3 ran")
        except ImportError:
            websocket = False
        if websocket:
            # pywsgi.WSGIServer(('', 3000), app, log=None, handler_class=WebSocketHandler).serve_forever()
            print("4 ran")
            log = logging.getLogger('werkzeug')
            log.disabled = True
            app.logger.disabled = True
        else:
            pywsgi.WSGIServer(('', 5001), app).serve_forever()
            print("5 ran")
    elif sio.async_mode == 'gevent_uwsgi':
        print('Start the application through the uwsgi server. Example:')
        print('uwsgi --http :5000 --gevent 1000 --http-websockets --master '
              '--wsgi-file app.py --callable app')
    else:
        print('Unknown async_mode: ' + sio.async_mode)
