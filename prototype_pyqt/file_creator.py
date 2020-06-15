import sys
import dicom2nifti

# Comprobación de seguridad, ejecutar sólo si se reciben 2 argumentos reales
if len(sys.argv) == 3:
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    print(input_directory)
    print(output_directory)
    try:
    	dicom2nifti.convert_directory(input_directory, output_directory)
    except:
        print()
    	pass