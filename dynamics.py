# The dynamics goes here.
# Inputs: Button presses (Q,W,O,and/or P)
# Outputs: New state of system


# Example of a class and making functions within it
class torso():
    def __init__(self,initial_x):
        self.X = initial_x
        return
    
    def x(self, x_state):
        self.X = x_state
        
    def X_and_Y(self, state, y_state):
        self.state = state
        
    def state(self):
        return self.X

