import sys
sys.path.append('..')
from common.model import Net
from common.layer import ParellelAffine, Affine, Sigmoid, SoftmaxWithLoss, Mean
class CustomCBOW(Net):
    def __init__(self, input_size, hidden_size, output_size, num):
        super().__init__()
        I, H, O = input_size, hidden_size, output_size
        self.add_layers([
            ParellelAffine(I, H, num),
            Mean(num),
            Affine(H, O)
        ])
        self.add_lossLayer([SoftmaxWithLoss()])
        self.word_vecs = self.get_params()[0]
    
    def get_inputw(self):
        params = self.get_params()
        return params[0], params[1]

    def get_outputw(self):
        params = self.get_params()
        return params[2], params[3]