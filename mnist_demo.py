import sys
import os

import numpy as np
import torch
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor

def poisson_generator(u_in,batch_size,dt,T):
    x_rand=np.random.uniform(0.0,1.0,(int(T/dt),batch_size,len(u_in)))
    input_data=(u_in*dt>x_rand)
    return torch.from_numpy(input_data)

def down_sample_spike(spike_data, dt, sliding_window, time_step, return_boolean=True):
    '''downsample the spike recordings'''
    x, _, _ = spike_data.shape
    result = np.array(
        [np.sum(spike_data[int(time_step / dt) * i:int(time_step / dt) * i + int(sliding_window / dt), :, :], axis=0)
         for i in range(int((x - sliding_window / dt) / (time_step / dt)))])
    if return_boolean:
        result = (result > 0)
    return result

def param_map(raw_dict,eps=1e-5):
    param_dict={}
    layer_1_scale=raw_dict['mlp.0.bn.bn_mean.weight']/(raw_dict['mlp.0.bn.bn_mean.running_var']+eps)**0.5
    param_dict['hidden_layer_weight']=raw_dict['mlp.0.linear.weight']*torch.reshape(layer_1_scale,(-1,1))
    param_dict['hidden_layer_weight']=param_dict['layer_1_weight'].float()
    param_dict['hidden_layer_I_ext_mean']=raw_dict['mlp.0.bn.bn_mean.bias']-raw_dict['mlp.0.bn.bn_mean.weight']*raw_dict['mlp.0.bn.bn_mean.running_mean']/(raw_dict['mlp.0.bn.bn_mean.running_var']+eps)**0.5
    param_dict['hidden_layer_I_ext_mean']=param_dict['hidden_layer_I_ext_mean'].float()
    #We set I_ext as constant current input without noise, so hidden_layer_I_ext_std should all be zero
    param_dict['hidden_layer_I_ext_std']=torch.zeros(len(param_dict['hidden_layer_I_ext_mean']))
    param_dict['decoding_layer_weight']=raw_dict['predict.weight']
    return param_dict

class MNIST_SNN():
    def __init__(self,save_dir,model_state_dict,config,
                 L=0.05,thresh=20.0,rest=0.0,reset=0.0,refrac=5,eps=1e-5,save_sliding_window=1.0,save_time_step=1.0):
        self.save_dir=save_dir
        self.model_state_dict=model_state_dict
        self.L=L
        self.thresh=thresh
        self.rest=rest
        self.reset=reset
        self.refrac=refrac
        self.eps=eps
        self.N_0=config['N_0']
        self.N_1=config['N_1']
        self.M=config['trial_number']
        self.time=config['simulation_time']
        self.batch_size=config['simulation_batch_size']
        self.dt=config['dt']
        self.decay_SNN = -self.dt / torch.log(torch.tensor([1 - L * self.dt]))
        self.save_sliding_window=save_sliding_window
        self.save_time_step=save_time_step

    def generate_noise_input(self,input_mean,input_std):
        timestep=int(self.time/self.dt)
        input_data=input_mean*self.dt+input_std*(self.dt**0.5)*torch.randn((timestep,self.batch_size,len(input_std)))
        return input_data

    def construct_and_run(self,u_in,indx):
        param_dict=self.model_state_dict
        network = Network(dt=self.dt, batch_size=self.batch_size, learning=False)
        input_layer = Input(n=self.N_0)
        network.add_layer(layer=input_layer, name="input")
        layer_1 = LIFNodes(n=self.N_1, thresh=self.thresh, rest=self.rest, reset=self.reset, sum_input=False, refrac=self.refrac-0.5*self.dt,tc_decay=self.decay_SNN)
        network.add_layer(layer=layer_1, name="layer_1")


        monitor_1 = Monitor(obj=layer_1,state_vars=("s"),time=int(self.time / self.dt))
        network.add_monitor(monitor=monitor_1, name="layer_1")


        connection_01 = Connection(source=input_layer, target=layer_1,w=torch.tensor(param_dict['hidden_layer_weight'].T))
        network.add_connection(connection=connection_01, source="input", target="layer_1")

        spike_layer_1=[]

        for _ in range(int(self.M / self.batch_size)):
            input_data_0=poisson_generator(u_in,self.batch_size,self.dt,self.time)

            input_data_1=self.generate_noise_input(param_dict['hidden_layer_I_ext_mean'],param_dict['hidden_layer_I_ext_std'])
            inputs = {"input": input_data_0,
                      'layer_1':input_data_1,
                      }
            # Simulate network on input data.
            network.run(inputs=inputs, time=self.time)
            spike_layer_1.append(down_sample_spike(monitor_1.get("s").numpy(),self.dt,self.save_sliding_window,self.save_time_step))
        spike_layer_1 = np.concatenate(spike_layer_1, axis=1)


        self.save_spk_data(spike_layer_1,indx,self.save_dir,'spike_layer_1','npy')

    def save_spk_data(self,spk_data,indx,save_dir,filename,save_type='npy'):
        if save_type=='npy':
            np.save(save_dir+filename+'_indx='+str(indx)+'.npy',spk_data)
        if save_type=='csv':
            np.savetxt(save_dir+filename+'_indx='+str(indx)+'.csv',spk_data,delimiter=',')
        return

def param_mapping_main(data_dir):
    Raw_model_state_dict=torch.load(data_dir+'RawMnn.pth')['net']
    param_dict=param_map(Raw_model_state_dict)
    torch.save(param_dict,data_dir+'trained_SNN_parameters.pt')

def mnist_SNN_main(data_dir,save_dir,config,indx):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_state_dict=torch.load(data_dir+'trained_SNN_parameters.pt')
    u_ins=np.load(data_dir+'input_images.npy')

    mnist_snn=MNIST_SNN(save_dir,model_state_dict,config)
    mnist_snn.construct_and_run(u_ins[indx,:],indx)

def mnist_decoder(data_dir,save_dir,config,sample_number):
    model_state_dict=torch.load(data_dir+'trained_SNN_parameters.pt')
    weight=model_state_dict['decoding_layer_weight'].detach().numpy()
    time=config['simulation_time']
    SNN_result=[]

    for i in range(sample_number):
        result=np.load(save_dir+'spike_layer_1_indx='+str(i)+'.npy')
        spk_mean=np.sum(result[:,0,:],axis=0)/time
        SNN_result.append(np.argmax(np.matmul(weight,spk_mean)))
        print(str(i)+' samples completed!')

    np.save(data_dir+'SNN_decoding.npy',np.array(SNN_result))

    target=np.load(data_dir+'target_labels.npy')[:sample_number]
    print('accuracy for ',sample_number,' : ',np.sum(SNN_result==target)/sample_number*100, ' %')

if __name__=='__main__':
    config={'N_0':784,
            'N_1':800,
            'trial_number':1000,
            'simulation_time':100,
            'simulation_batch_size':100,
            'dt':0.025}
    data_dir='mnist_file/'
    save_dir=data_dir+'SNN/'
    sample_number=2

    #####################################################################
    #Parametric mapping
    param_mapping_main(data_dir)

    ######################################################################
    #SNN simulation
    #indx=int(sys.argv[1])
    for indx in range(sample_number):
        mnist_SNN_main(data_dir,save_dir,config,indx)

    ######################################################################
    #decoding
    mnist_decoder(data_dir,save_dir,config,sample_number)