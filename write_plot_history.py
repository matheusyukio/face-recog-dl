import numpy as np
# visualizazao
import matplotlib.pyplot as plt

def write_results(filename, acc, loss, history):
    VALIDATION_ACCURACY = acc
    VALIDATION_LOSS = loss
    HISTORY = history
    file = open(filename, 'a+')
    file.write('VALIDATION_ACCURACY \n')
    file.write(str(VALIDATION_ACCURACY))
    file.write('\n')
    file.write('VALIDATION_ACCURACY mean\n')
    file.write(str(np.mean(VALIDATION_ACCURACY)))
    file.write('\n')
    file.write('VALIDATION_ACCURACY std\n')
    file.write(str(np.std(VALIDATION_ACCURACY)))
    file.write('\n')
    file.write('\n')
    file.write('VALIDATION_LOSS \n')
    file.write(str(VALIDATION_LOSS))
    file.write('\n\n')
    for hist in range(len(HISTORY)):
        file.write('VALIDATION_ACCURACY HISTORY ' + str(hist) + '\n')
        file.write(str(VALIDATION_ACCURACY[hist]))
        file.write('\n')
        file.write('VALIDATION_LOSS HISTORY ' + str(hist) + '\n')
        file.write(str(VALIDATION_LOSS[hist]))
        file.write('\n')
        file.write('HISTORY ' + str(hist) + ' \n')
        file.write('EPOCHS ' + str(len(HISTORY[hist].history['loss'])) + ' \n')
        file.write(str(HISTORY[hist].history))
        file.write('\n\n')
    file.close()
    plot_train_test_loss(HISTORY[hist].history, filename)

def plot_train_test_loss(history, filename):
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title(filename + ' modelo acc')
    plt.ylabel('acc')
    plt.xlabel('epoca')
    plt.legend(['treino', 'validacao'], loc='upper left')
    #plt.show()
    plt.savefig(filename + 'ACC.png')
    plt.close()

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title(filename +  ' modelo loss')
    plt.ylabel('loss')
    plt.xlabel('epoca')
    plt.legend(['treino', 'validacao'], loc='upper left')
    #plt.show()
    plt.savefig(filename + 'LOSS.png')
    plt.close()