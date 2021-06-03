class Config:
    def __init__(self):
        self.train_X_path='../data/base_data/trainX'
        self.train_Y_path='../data/base_data/trainY'
        self.val_X_path='../data/base_data/testX'
        self.val_Y_path='../data/base_data/testY'
        self.val_save_path='./generated_img'
        self.batch_size=1
        self.lr=2e-4
        self.lambda_identity=0.0
        self.lambda_cycle=10
        self.lambda_dis=0.5
        self.epochs=200
        self.load_model=True
        self.ckpt_path='./save/train'
