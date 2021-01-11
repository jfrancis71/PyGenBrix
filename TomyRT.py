import time
import random
import numpy as np
import rpyc
import torch
import torch.optim as optim

from PyGenBrix import DataSetUtils as ds_utils

#my_tomy = TomyRT.TomyRT()
#my_tomy.main()

class TomyRT():
    def __init__( self ):
        self.conn = rpyc.classic.connect("192.168.0.9" )
        self.conn._config['sync_request_timeout'] = 100000
        self.tomy = self.conn.modules['PyGenBrix.TomyRT_ev3dev2_server']
        self.replay_buffer = []

        self.qnet = torch.nn.Sequential(
            torch.nn.Conv2d( 3, 16, 5, stride=1, padding=2 ), torch.nn.Tanh(),
            torch.nn.Conv2d( 16, 32, 5, stride=2, padding=2 ), torch.nn.Tanh(),
            torch.nn.Conv2d( 32, 32, 5, stride=2, padding=2 ), torch.nn.Tanh(),
            torch.nn.Conv2d( 32, 32, 5, stride=2, padding=2 ), torch.nn.Tanh(),
            torch.nn.Conv2d( 32, 32, 5, stride=2, padding=2 ), torch.nn.Tanh(),
            torch.nn.Conv2d( 32, 4, 5, stride=2, padding=2 ), torch.nn.Tanh(),
            torch.nn.Flatten()
)
        self.optimizer = optim.SGD( self.qnet.parameters(), lr=0.01 )

    def reward( self, previous_action, col ):
        reward = 0.0
        crashes = 0
        if previous_action >= 2:
            reward += .1
        if previous_action == 0 or previous_action == 2:
            reward += .03
        if col == 5 or col == 7:
            reward += -1.0
            print( "CRASH" )
            crashes = 1
        return ( reward, crashes )

    def add_replay_buffer( self, time, previous_image, current_image, previous_action, color ):
        reward = self.reward( previous_action, color )[0]
    
        self.replay_buffer.append( (
            time,
            previous_image,
            current_image, 
            previous_action,
            reward ) )

    def training_step( self ):
    
        if ( len( self.replay_buffer ) < 16 ):
            return
    
        dats = random.choices( self.replay_buffer, k = 16 )
    
        self.optimizer.zero_grad()
        tensor_images = torch.tensor( [ dat[1] for dat in dats ] )
        tensor_next_images = torch.tensor( [ dat[2] for dat in dats ] )
        tensor_actions = torch.tensor( [ dat[3] for dat in dats ] )
        tensor_rewards = torch.tensor( [ dat[4] for dat in dats ] )

        q = self.qnet( tensor_images )[range(16),tensor_actions]
        q1 = 0.9 * ( tensor_rewards + torch.max( self.qnet( tensor_next_images ), dim = 1 ).values )
        lossT = ( q - q1 ) * ( q - q1 )
        loss = torch.mean( lossT )
        loss.backward()
        self.optimizer.step()


    def training_steps( self ):
        for x in range(1500):
            self.training_step()

    def exec_step( self, current_image, epsilon = 0.1 ):
        tensor_image = torch.tensor( [ current_image ] )
        net_output = self.qnet( tensor_image )
        print( len( self.replay_buffer ), net_output )
        action = net_output[0].argmax().cpu().detach().numpy().item()
        if ( np.random.random_sample() > 1.0-epsilon ):
            action = np.random.randint( low = 0, high = 4 )
            print( " RANDOM ACTION" )
        color = -1
        if ( action == 0 ):
            color = self.tomy.tank( 0, -1 )
        else:
            if ( action == 1 ):
                color = self.tomy.tank( 0, 1 )
            else:
                if ( action == 2 ):
                    color = self.tomy.tank( 1, -1 )
                else:
                    color = self.tomy.tank( 1, 1 )

        return action, color

    def main( self ):
        for t in range(50):
            for f in range(100):
                current_image = np.transpose( np.array( ds_utils.import_camera_image().resize( ( 32, 32 ) ) ).astype( np.float32 ), [ 2, 0, 1 ] )/255.
                ( current_action, color ) = self.exec_step( current_image )
                if f > 0:
                    self.add_replay_buffer( time.time(), previous_image, current_image, previous_action, color )
                time.sleep( 0.6 )
                previous_action = current_action
                previous_image = current_image
            self.tomy.tank_stop()
            self.training_steps()
            print( "   VALIDATION" )
            self.main_drive( 100 )

    def main_drive( self, steps = 1000 ):
        reward = 0.0
        crashes = 0
        for f in range( steps ):
            current_image = np.transpose( np.array( ds_utils.import_camera_image().resize( ( 32, 32 ) ) ).astype( np.float32 ), [ 2, 0, 1 ] )/255.
            ( current_action, color ) = self.exec_step( current_image, epsilon = 0.0 )
            if f > 0:
                ( current_reward, current_crashes ) = self.reward( previous_action, color )
                reward += current_reward
                crashes += current_crashes
            time.sleep( 0.6 )
            previous_action = current_action
        print( "   Average reward = ", reward/(steps-1), " crashes = ", crashes )

