class ShallowNetEncoder(nn.Module):
    """
    Pytorch implementation of the ShallowNet Encoder.

    See ShallowNet for some references.
    The expected **input** is a **3D tensor** with size
    (Batch x Channels x Samples).

    https://github.com/MedMaxLab/selfEEG/blob/024402ba4bde95051d86ab2524cc71105bfd5c25/selfeeg/models/zoo.py#L693

    Parameters
    ----------
    Chans: int
        The number of EEG channels.
    F: int, optional
        The number of output filters in the temporal convolution layer.

        Default = 40
    K1: int, optional
        The length of the temporal convolutional layer.

        Default = 25
    Pool: int, optional
        The temporal pooling kernel size.

        Default = 75
    p: float, optional
        Dropout probability. Must be in [0,1)

        Default= 0.2

    Note
    ----
    In this implementation, the number of channels is an argument.
    However, in the original paper authors preprocess EEG data by
    selecting a subset of only 21 channels. Since the net is very
    minimalistic, please follow the authors' notes.


    """

    def __init__(self, samples, Chans, F=40, K1=25, Pool=75, p=0.2, num_extracted_features=100):

        super(ShallowNetEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, F, (1, K1), stride=(1, 1))
        self.conv2 = nn.Conv2d(F, F, (Chans, 1), stride=(1, 1))
        self.batch1 = nn.BatchNorm2d(F)
        self.pool2 = nn.AvgPool2d((1, Pool), stride=(1, 15))
        self.flatten2 = nn.Flatten()
        self.drop1 = nn.Dropout(p)
        self.lin = nn.Linear(
            F * ((samples - K1 + 1 - Pool) // 15 + 1), num_extracted_features
        )

    def forward(self, x):
        """
        :meta private:
        """
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batch1(x)
        x = torch.square(x)
        x = self.pool2(x)
        x = torch.log(torch.clamp(x, 1e-7, 10000))
        x = self.flatten2(x)
        x = self.drop1(x)
        x = self.lin(x)

        return x