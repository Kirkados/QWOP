"""
This script saves and loads network parameters 
"""

import os
import tensorflow as tf

from settings import Settings

class Saver:
    
    def __init__(self, sess):
        self.sess = sess
        
    def save(self, n_episode):
        # Save all the tensorflow parameters from this session into a file
        # The file is saved to the directory Settings.MODEL_SAVE_DIRECTORY.
        # It uses the n_episode in the file name
        
        print("Saving models at episode number " + str(n_episode) + "...")
        
        os.makedirs(os.path.dirname(Settings.MODEL_SAVE_DIRECTORY), exist_ok = True)
        self.saver.save(self.sess, Settings.MODEL_SAVE_DIRECTORY + "Episode_" + str(n_episode) + ".ckpt")
        
        print("Model saved!")
        
        
    def load(self):
        # Try to load in weights to the networks in the current Session.
        # If it fails, or we don't want to load (Settings.TRY_TO_LOAD_IN_PARAMETERS = False)
        # then we start from scratch
        
        self.saver = tf.train.Saver() # initialize the tensorflow Saver()
        
        if Settings.TRY_TO_LOAD_IN_PARAMETERS:
            
            print("Loading in previously-trained model")
            
            try:
                
                ckpt = tf.train.get_checkpoint_state(Settings.MODEL_SAVE_DIRECTORY)
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                
                print("Model successfully loaded!")
                return True
            
            except (ValueError, AttributeError):
                
                print("No model found... re-initializing all parameters!")
                return False
            
        else:
            
            return False