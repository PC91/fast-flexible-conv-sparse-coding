import numpy as np

def write_file(filename, array, dim):
    f = open(filename, 'w')
    if (dim == 2):
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                f.write('%f ' % array[i,j])
            f.write('\n')
            
    elif (dim == 3):
        for k in range(array.shape[0]):        
            for i in range(array.shape[1]):
                for j in range(array.shape[2]):
                    f.write('%f ' % array[k,i,j])
                f.write('\n')
            f.write('\n')
    elif (dim == 4):
        for k in range(array.shape[0]):
            for l in range(array.shape[1]):
                for i in range(array.shape[2]):
                    for j in range(array.shape[3]):
                        f.write('%f ' % array[k,l,i,j])
                    f.write('\n')
                f.write('\n')
            f.write('\n')
    f.close()



def read_file(filename, array, dim):
    f = np.loadtxt(filename)
    line = 0
    if (dim == 2):
        for i in range(array.shape[0]):
            array[i,:] = f[line]
            line += 1
    elif (dim == 3):
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                array[i,j,:] = f[line]
                line += 1
    elif (dim == 4):
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                for k in range(array.shape[2]):
                    array[i,j,k,:] = f[line]
                    line += 1