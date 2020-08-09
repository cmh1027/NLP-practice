import sys
sys.path.append('..')  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.model import Net
from common.layer import Affine, Sigmoid, SoftmaxWithLoss
class TwoLayerNet(Net):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        I, H, O = input_size, hidden_size, output_size

        # 계층 생성
        self.add_layers([
            Affine(I, H),
            Sigmoid(),
            Affine(H, O)
        ])
        self.loss_layer = SoftmaxWithLoss()