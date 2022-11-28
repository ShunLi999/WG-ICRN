

class ConvBNRelu(Model):
    def __init__(self, ch, kernelsz=3, strides=1, padding='same'):
        super(ConvBNRelu, self).__init__()
        self.model = tf.keras.models.Sequential([
            Conv2D(ch, kernelsz, strides=strides, padding=padding),
            BatchNormalization(),
            Activation('relu')
        ])

    def call(self, x):
        x = self.model(x, training=False)
        return x


class InceptionBlk(Model):
    def __init__(self, ch, strides=1):
        super(InceptionBlk, self).__init__()
        self.ch = ch
        self.strides = strides
      #  self.c1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
        self.c1_1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
        self.c1_2 = ConvBNRelu(ch, kernelsz=3, strides=1)
        self.c1_3 = ConvBNRelu(ch, kernelsz=3, strides=1)

        self.c2_1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
        self.c2_2 = ConvBNRelu(ch, kernelsz=3, strides=1)
        self.c2_3 = ConvBNRelu(ch, kernelsz=3, strides=1)
        self.c2_4 = ConvBNRelu(ch, kernelsz=3, strides=1)

        self.p3_1 = MaxPool2D(3, strides=1, padding='same')
        self.c3_2 = ConvBNRelu(ch, kernelsz=1, strides=strides)

    def call(self, x):
        x1_1 = self.c1(x)
        x1_2 = self.c1_2(x1_1)
        x1_3 = self.c1_3(x1_2)
        x2_1 = self.c2_1(x)
        x2_2 = self.c2_2(x2_1)
        x2_3 = self.c2_3(x2_2)
        x2_4 = self.c2_4(x2_3)
        x3_1 = self.c3_1(x)
        x3_2 = self.c3_2(x3_1)

        #x4_1 = self.p4_1(x)
        #x4_2 = self.c4_2(x4_1)
        # concat along axis=channel
        x = tf.concat([x1_3 ,x2_4, x3_2], axis=3)
        return x


class Inception(Model):
    def __init__(self, num_blocks, num_classes, init_ch=16, **kwargs):
        super(Inception, self).__init__(**kwargs)
        self.in_channels = init_ch
        self.out_channels = init_ch
        self.num_blocks = num_blocks
        self.init_ch = init_ch
        self.c1 = ConvBNRelu(init_ch)
        self.blocks = tf.keras.models.Sequential()
        for block_id in range(num_blocks):
            for layer_id in range(2):
                if layer_id == 0:
                    block = InceptionBlk(self.out_channels, strides=2)
                else:
                    block = InceptionBlk(self.out_channels, strides=1)
                self.blocks.add(block)
            # enlarger out_channels per block
            self.out_channels *= 2
        self.p1 = GlobalAveragePooling2D()
        self.f1 = Dense(num_classes, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y

model = Inception(num_blocks=4, num_classes=8)


#
