from nipype.interfaces.ants import Registration, RegistrationSynQuick
import ants
import os


class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

def generate_non_rigid_registration_with_mni(path,patient):
    typeFile = '_12' 
    if patient < 20: dir = 'Utrecht'
    elif patient < 40: dir = 'Singapore'
    else: dir = 'GE3T'
    reg = Registration()
    
    reg.inputs.fixed_image = '../../Data/MNI_related/MNI152_T1_1mm.nii.gz'
    reg.inputs.moving_image = path
    reg.inputs.transforms = ['Affine', 'SyN']
    reg.inputs.metric = ['Mattes']*2
    reg.inputs.metric_weight = [1]*2 # Default (value ignored currently by ANTs)
    reg.inputs.smoothing_sigmas = [[1,0], [2,1,0]]
    reg.inputs.transform_parameters = [(2.0,), (0.25, 3.0, 0.0)]
    reg.inputs.number_of_iterations = [[1500, 200], [100, 50, 30]]
    reg.inputs.shrink_factors = [[2,1], [3,2,1]]
    reg.inputs.output_warped_image = str(patient) + '_' + dir + '_12.nii.gz'
    reg.inputs.output_inverse_warped_image = str(patient) + '_' + dir + '_12_inversed.nii.gz'


    
    res = reg.run()
    print(res.outputs)
    
def generate_antsSyN(path, patient):
    typeFile = '_13' 
    if patient < 20: dir = 'Utrecht'
    elif patient < 40: dir = 'Singapore'
    else: dir = 'GE3T'
    model_path = dir + '_' + str(patient)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    
    
    with cd(model_path):
        reg = RegistrationSynQuick()
        reg.inputs.fixed_image = '../../Data/MNI_related/MNI152_T1_1mm.nii.gz'
        reg.inputs.moving_image = '../' + path
        reg.inputs.num_threads = 2
        reg.inputs.dimension = 3
        res = reg.run()
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
        generate_antsSyN(T1_image, patient)



if __name__=='__main__':
    main()
