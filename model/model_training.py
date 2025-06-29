import sys
import os
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, matthews_corrcoef, accuracy_score
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from Bio import SeqIO
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau

np.random.seed(42)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_features():
    # train_pos_fm = np.load('full_train_pos_features.npy')
    # train_neg_fm = np.load('full_train_neg_features.npy')
    # test_pos_fm = np.load('full_test_pos_features.npy')
    # test_neg_fm = np.load('full_test_neg_features.npy')

    train_pos_fm = np.load('train_pos_features.npy')
    train_neg_fm = np.load('train_neg_features.npy')
    test_pos_fm = np.load('test_pos_features.npy')
    test_neg_fm = np.load('test_neg_features.npy')
    print(f"RNA-FM特征维度：训练正样本 {train_pos_fm.shape}，训练负样本 {train_neg_fm.shape}")
    print(f"RNA-FM特征维度：测试正样本 {test_pos_fm.shape}，测试负样本 {test_neg_fm.shape}")
    


    # 加载原始序列
    def load_fasta(path):
        return [str(seq.seq) for seq in SeqIO.parse(path, "fasta")]

    train_pos_seq = load_fasta('Mature_mRNA-train-Pos.txt')
    train_neg_seq = load_fasta('Mature_mRNA-train-Neg.txt')
    test_pos_seq = load_fasta('Mature_mRNA-test-Pos.txt')
    test_neg_seq = load_fasta('Mature_mRNA-test-Neg.txt')

    # train_pos_seq = load_fasta('Fulltranscriptmode-train-Pos.txt')
    # train_neg_seq = load_fasta('Fulltranscriptmode-train-Neg.txt')
    # test_pos_seq = load_fasta('Full-transcript-mode-test-Pos.txt')
    # test_neg_seq = load_fasta('Full-transcript-mode-test-Neg.txt')


    return (train_pos_fm, train_neg_fm, test_pos_fm, test_neg_fm,
            train_pos_seq, train_neg_seq, test_pos_seq, test_neg_seq)


def onehot(seqs, seq_length=43):
    bases = ['A', 'C', 'G', 'U']
    X = np.zeros((len(seqs), seq_length, len(bases)))
    for i, seq in enumerate(seqs):
        processed_seq = seq.ljust(seq_length)[:seq_length]
        for j, base in enumerate(processed_seq):
            if base in bases:
                X[i, j, bases.index(base)] = 1
    return X


def prepare_data():
    (train_pos_fm, train_neg_fm, test_pos_fm, test_neg_fm,
     train_pos_seq, train_neg_seq, test_pos_seq, test_neg_seq) = load_features()


    min_train_pos = min(len(train_pos_fm), len(train_pos_seq))
    min_train_neg = min(len(train_neg_fm), len(train_neg_seq))
    min_test_pos = min(len(test_pos_fm), len(test_pos_seq))
    min_test_neg = min(len(test_neg_fm), len(test_neg_seq))


    X_train_fm = np.vstack([train_pos_fm[:min_train_pos], train_neg_fm[:min_train_neg]])
    X_train_seq = np.vstack([onehot(train_pos_seq[:min_train_pos]),
                             onehot(train_neg_seq[:min_train_neg])])
    y_train = np.array([1] * min_train_pos + [0] * min_train_neg)

    X_test_fm = np.vstack([test_pos_fm[:min_test_pos], test_neg_fm[:min_test_neg]])
    X_test_seq = np.vstack([onehot(test_pos_seq[:min_test_pos]),
                            onehot(test_neg_seq[:min_test_neg])])
    y_test = np.array([1] * min_test_pos + [0] * min_test_neg)


    indices = np.random.permutation(len(y_test))
    return (X_train_seq, X_train_fm, y_train,
            X_test_seq[indices], X_test_fm[indices], y_test[indices])


def build_model(seq_shape=(43, 4), fm_shape=640):

    seq_input = Input(shape=seq_shape)


    x = Conv1D(512, 9, activation='relu', padding='same', kernel_regularizer=l2(1e-4))(seq_input)
    x = Conv1D(512, 9, activation='relu', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = Conv1D(512, 9, activation='relu', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.2)(x)


    x = GlobalMaxPooling1D()(x)
    x = Dense(512, activation='relu')(x)


    fm_input = Input(shape=(fm_shape,))
    y = Dense(1024, activation='relu', kernel_regularizer=l2(1e-4))(fm_input)
    y = Dropout(0.2)(y)


    combined = Concatenate()([x, y])


    z = Dense(1536, activation='relu')(combined)
    z = Dropout(0.2)(z)
    output = Dense(1, activation='sigmoid')(z)

    model = Model(inputs=[seq_input, fm_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train_and_evaluate():

    X_train_seq, X_train_fm, y_train, X_test_seq, X_test_fm, y_test = prepare_data()


    fold_models = []
    val_metrics = {'auc': [], 'acc': [], 'f1': [], 'mcc': [], 'sn': [], 'sp': []}


    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_seq, y_train)):
        print(f"\n========== Fold {fold + 1} ==========")


        X_tr_seq, X_val_seq = X_train_seq[train_idx], X_train_seq[val_idx]
        X_tr_fm, X_val_fm = X_train_fm[train_idx], X_train_fm[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]


        model = build_model()
        callbacks = [
            EarlyStopping(patience=20, monitor='val_loss', restore_best_weights=True),
            ModelCheckpoint(f'fold{fold}_model.h5', save_best_only=True)
        ]


        model.fit(
            [X_tr_seq, X_tr_fm], y_tr,
            validation_data=([X_val_seq, X_val_fm], y_val),
            epochs=200,
            batch_size=64,
            class_weight=dict(
                enumerate(class_weight.compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr))),
            callbacks=callbacks,
            verbose=1
        )


        model.load_weights(f'fold{fold}_model.h5')
        y_pred = model.predict([X_val_seq, X_val_fm]).flatten()
        y_pred_bin = (y_pred > 0.5).astype(int)


        val_metrics['auc'].append(roc_auc_score(y_val, y_pred))
        val_metrics['acc'].append(accuracy_score(y_val, y_pred_bin))
        val_metrics['f1'].append(f1_score(y_val, y_pred_bin))
        val_metrics['mcc'].append(matthews_corrcoef(y_val, y_pred_bin))


        tn, fp, fn, tp = confusion_matrix(y_val, y_pred_bin).ravel()
        sn = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        sp = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        val_metrics['sn'].append(sn)
        val_metrics['sp'].append(sp)


        fold_models.append(model)


    print("\nValidation Metrics:")
    print(f"AUC: {np.nanmean(val_metrics['auc']):.4f} ± {np.nanstd(val_metrics['auc']):.4f}")
    print(f"Accuracy: {np.nanmean(val_metrics['acc']):.4f} ± {np.nanstd(val_metrics['acc']):.4f}")
    print(f"F1: {np.nanmean(val_metrics['f1']):.4f} ± {np.nanstd(val_metrics['f1']):.4f}")
    print(f"MCC: {np.nanmean(val_metrics['mcc']):.4f} ± {np.nanstd(val_metrics['mcc']):.4f}")
    print(f"SN: {np.nanmean(val_metrics['sn']):.4f} ± {np.nanstd(val_metrics['sn']):.4f}")
    print(f"SP: {np.nanmean(val_metrics['sp']):.4f} ± {np.nanstd(val_metrics['sp']):.4f}")


    test_preds = []
    for model in fold_models:
        test_preds.append(model.predict([X_test_seq, X_test_fm]))
    y_test_pred = np.mean(test_preds, axis=0).flatten()
    y_test_bin = (y_test_pred > 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_test_bin).ravel()
    sn = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    sp = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    test_metrics = {
        'AUC': roc_auc_score(y_test, y_test_pred),
        'Accuracy': accuracy_score(y_test, y_test_bin),
        'F1': f1_score(y_test, y_test_bin),
        'MCC': matthews_corrcoef(y_test, y_test_bin),
        'SN': sn,
        'SP': sp
    }


    print("\nTest Metrics with Ensemble:")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")




if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:

            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    train_and_evaluate()



