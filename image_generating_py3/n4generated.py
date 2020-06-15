from nipype.interfaces.ants import N4BiasFieldCorrection
import os

def generateN4Bias(path, patient, type='T1'):
    print(path)
    typeFile = '_10' if type == 'T1' else '_11'
    if patient < 20: dir = 'Utrecht'
    elif patient < 40: dir = 'Singapore'
    else: dir = 'GE3T'
    n4 = N4BiasFieldCorrection()
    n4.inputs.dimension = 3
    n4.inputs.input_image = path
    n4.inputs.output_image= str(patient) + '_' + dir + typeFile + '.nii.gz'
    n4.inputs.bias_image = str(patient) + '_' + dir + typeFile + '_biasImage.nii.gz'
    res = n4.run()
    print(res.outputs)


#leave-one-out evaluation
def main():
    patient_num  = 60
    for patient in range(0, patient_num):
        if patient < 20: dir = '../../Data/originalData/Utrecht/'
        elif patient < 40: dir = '../../Data/originalData/Singapore/'
        else: dir = '../../Data/originalData/GE3T/'
        dirs = os.listdir(dir)
        dirs.sort()
        dir += dirs[patient%20]
        T1_image = dir + '/orig/T1.nii.gz'
        FLAIR_image = dir + '/orig/FLAIR.nii.gz'
        generateN4Bias(T1_image, patient, type="T1")
        generateN4Bias(FLAIR_image, patient, type="Flair" )



if __name__=='__main__':
    main()
