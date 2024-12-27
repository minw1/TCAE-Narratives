import nibabel.freesurfer.io as fsio

# Load the .annot file
annot_file = "/home/wsm32/project/wsm_thesis_scratch/narratives/fs6_transfer/label/lh.aparc.a2009s.annot"
labels, ctab, names = fsio.read_annot(annot_file)
for i, name in enumerate(names):
    print(f"{i}: {name.decode('utf-8')}")
