from models import AutoIN
import torch

if __name__ == '__main__':
    cal_size = 4
    values_size = 10
    embeding_size = 256

    cal_num = 2
    value_num = 3

    model = AutoIN(cal_size, values_size, embeding_size, cal_num, value_num, dropout=0.5)
    # print(model)
    cal_index = torch.LongTensor([1, 3]).unsqueeze(0)
    value_data = torch.Tensor([[0.5], [0.1], [0.1]]).unsqueeze(0)
    value_index = torch.LongTensor([0, 1, 5]).unsqueeze(0)
    # check
    assert cal_index.size(1) == cal_num
    assert value_data.size(1) == value_num
    assert value_index.size(1) == value_num

    r = model(cal_index, value_data, value_index)
    print(r)
