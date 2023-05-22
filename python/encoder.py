from aclnet import Net

class Encoder:
    def __init__(self, args):
        self.device_id = args['device_id']
        self.encoder_input_shape = args['encoder_input_shape']
        self.encoder_output_shape = args['encoder_output_shape']
        self.encoder_om_path = args['encoder_om_path']
        self.model = Net(self.device_id, 
                        self.encoder_input_shape, 
                        self.encoder_output_shape, 
                        self.encoder_om_path)
    
    def forward(self, speechs, speech_lens):
        result = self.model.run([speechs, speech_lens])
        return result
