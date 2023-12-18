
class Environment:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.data_iter = iter(data_loader)
        self.current_data = None
        self.num_of_batches = 7000//512

    def reset(self):
        # 重置数据加载器的迭代器
        self.data_iter = iter(self.data_loader)
        self.current_data = next(self.data_iter, None)

    def step(self):
        # 获取下一个数据批次
        if self.current_data is None:
            done = True
            next_data = None
        else:
            done = False
            next_data = self.current_data
            self.current_data = next(self.data_iter, None)
        return next_data, done

    # 如果还需要索引，可以添加如下方法
    # 但注意，DataLoader不保证数据的顺序，特别是在shuffle=True时
    def get_current_index(self):
        raise NotImplementedError("DataLoader does not support index tracking.")
