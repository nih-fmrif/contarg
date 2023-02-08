# ConTarg

A python package implementing/wrapping methods for functional connectivity personalized targeting of rTMS therapy.

## Getting test data
```commandline
datalad clone ///openneuro/ds002330
cd ds002330/sub-02
datalad get -r .
```


## Example run
```commandline
contarg hierarchical run \
--bids-dir=data/ds002330 \
--derivatives-dir=data/derivatives/ \
--database-file=data/pybids_0.15.2_db \
--run-name=firsttest \
--space=T1w \
--smoothing-fwhm=3 \
--ndummy=5 \
--tr=1.9 \
--subject=02 \
--run=1 \
--njobs=2


```

## ROIs
Currently we just have spherical ROIs based on Cash et al., 2022 for defining SGC and DLPFC.
They were created as follows.
```commandline
# make SGC mask
3dcalc -a /usr/local/apps/fsl/6.0.4/data/standard/MNI152_T1_2mm.nii.gz   \
       -expr 'step(100-(x+6)*(x+6)-(y+16)*(y+16)-(z+10)*(z+10))' \
       -prefix SGCsphere_space-MNI152NLin6Asym_res-02.nii.gz
# make blSGC mask
3dcalc -a /usr/local/apps/fsl/6.0.4/data/standard/MNI152_T1_2mm.nii.gz   \
       -expr 'step(100-(x-6)*(x-6)-(y+16)*(y+16)-(z+10)*(z+10))' \
       -prefix leftSGCsphere_space-MNI152NLin6Asym_res-02.nii.gz
3dcalc -a SGCsphere_space-MNI152NLin6Asym_res-02.nii.gz   \
       -b leftSGCsphere_space-MNI152NLin6Asym_res-02.nii.gz    \
       -expr 'step(a+b)' \
       -prefix bilateralSGCspheres_space-MNI152NLin6Asym_res-02.nii.gz
# BA9 20mm -36, 39, 43
3dcalc -a /usr/local/apps/fsl/6.0.4/data/standard/MNI152_T1_2mm.nii.gz   \
       -expr 'step(400-(x-36)*(x-36)-(y+39)*(y+39)-(z-43)*(z-43))' \
       -prefix BA9sphere_space-MNI152NLin6Asym_res-02.nii.gz

# BA46 20mm -44, 40, 29
3dcalc -a /usr/local/apps/fsl/6.0.4/data/standard/MNI152_T1_2mm.nii.gz   \
       -expr 'step(400-(x-44)*(x-44)-(y+40)*(y+40)-(z-29)*(z-29))' \
       -prefix BA46sphere_space-MNI152NLin6Asym_res-02.nii.gz

# 5cm 20mm -41, 16, 54
3dcalc -a /usr/local/apps/fsl/6.0.4/data/standard/MNI152_T1_2mm.nii.gz   \
       -expr 'step(400-(x-41)*(x-41)-(y+16)*(y+16)-(z-54)*(z-54))' \
       -prefix 5cmsphere_space-MNI152NLin6Asym_res-02.nii.gz

# F3 20mm -37, 26, 49
3dcalc -a /usr/local/apps/fsl/6.0.4/data/standard/MNI152_T1_2mm.nii.gz   \
       -expr 'step(400-(x-37)*(x-37)-(y+26)*(y+26)-(z-49)*(z-49))' \
       -prefix F3sphere_space-MNI152NLin6Asym_res-02.nii.gz

# DLPFC
3dcalc -a BA9sphere_space-MNI152NLin6Asym_res-02.nii.gz -b BA46sphere_space-MNI152NLin6Asym_res-02.nii.gz \
       -c 5cmsphere_space-MNI152NLin6Asym_res-02.nii.gz -d F3sphere_space-MNI152NLin6Asym_res-02.nii.gz  \
       -expr 'step(a + b + c + d)'  \
       -prefix DLPFCspheres_space-MNI152NLin6Asym_res-02.nii.gz

# DLPFC + brainmask
3dcalc -a BA9sphere_space-MNI152NLin6Asym_res-02.nii.gz -b BA46sphere_space-MNI152NLin6Asym_res-02.nii.gz \
       -c 5cmsphere_space-MNI152NLin6Asym_res-02.nii.gz -d F3sphere_space-MNI152NLin6Asym_res-02.nii.gz  \
       -e /usr/local/apps/fsl/6.0.4/data/standard/MNI152_T1_2mm_brain_mask.nii.gz \
       -expr 'and(step(a + b + c + d),e)'  \
       -prefix DLPFCspheresmasked_space-MNI152NLin6Asym_res-02.nii.gz
       
# BA46 + brainmask
3dcalc -a /usr/local/apps/fsl/6.0.4/data/standard/MNI152_T1_2mm.nii.gz   \
       -b /usr/local/apps/fsl/6.0.4/data/standard/MNI152_T1_2mm_brain_mask.nii.gz \
       -expr 'and(a,b)' \
       -prefix BA46spheremasked_space-MNI152NLin6Asym_res-02.nii.gz
```


This is a Python project packaged according to [Contemporary Python Packaging - 2023][].

[Contemporary Python Packaging - 2023]: https://effigies.gitlab.io/posts/python-packaging-2023/
