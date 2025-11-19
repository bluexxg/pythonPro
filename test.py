import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt


# 数据增强策略 - 对小型数据集至关重要
def create_data_augmentation():
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])


# 加载小型数据集示例 (CIFAR-10)
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# 标准化像素值
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 转换为分类格式
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print(f"训练集形状: {x_train.shape}")
print(f"测试集形状: {x_test.shape}")


def build_small_cnn(input_shape=(32, 32, 3), num_classes=10):
    net = keras.Sequential([
        # 数据增强层
        create_data_augmentation(),

        # 第一卷积块
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # 第二卷积块
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # 第三卷积块
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # 全连接层
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return net


# 构建模型
model = build_small_cnn()
model.summary()


def train_model_with_callbacks(model, x_train, y_train, x_test, y_test):
    # 编译模型
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 回调函数
    callbacks = [
        # 早停防止过拟合
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        ),
        # 学习率调度
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=10,
            min_lr=1e-7
        ),
        # 模型检查点
        keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]

    # 训练模型
    history = model.fit(
        x_train, y_train,
        batch_size=32,
        epochs=10,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    return history


# 开始训练
history = train_model_with_callbacks(model, x_train, y_train, x_test, y_test)


def evaluate_and_visualize(model, history, x_test, y_test):
    # 评估模型
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"测试准确率: {test_accuracy:.4f}")
    print(f"测试损失: {test_loss:.4f}")

    # 绘制训练历史
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 准确率曲线
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # 损失曲线
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # 预测示例
    predictions = model.predict(x_test[:10])
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test[:10], axis=1)

    print("预测示例:")
    for i in range(5):
        print(f"真实: {true_classes[i]}, 预测: {predicted_classes[i]}")


evaluate_and_visualize(model, history, x_test, y_test)


def build_compact_cnn(input_shape=(32, 32, 3), num_classes=10):
    """更紧凑的模型，适合超小数据集"""
    model = keras.Sequential([
        create_data_augmentation(),

        # 使用深度可分离卷积减少参数
        layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same',
                               input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.SeparableConv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),  # 使用全局平均池化代替全连接层
        layers.Dropout(0.4),

        layers.Dense(num_classes, activation='softmax')
    ])

    return model


# 对于非常小的数据集使用紧凑模型
compact_model = build_compact_cnn()
compact_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
