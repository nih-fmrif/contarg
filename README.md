# ConTarg

A python package implementing/wrapping methods for functional connectivity personalized targeting of rTMS therapy.

## Getting test data
```commandline
git submodule init
git submodule update
datalad get  contarg/test/data/derivatives/fmriprep/sub-*/anat/sub-*_from-MNI152NLin6Asym_to-T1w_mode-image_xfm.h5 \ 
            contarg/test/data/derivatives/fmriprep/sub-*/anat/*preproc_T1w.* \
            contarg/test/data/derivatives/fmriprep/sub-*/func/*preproc_bold.* \
            contarg/test/data/derivatives/fmriprep/sub-*/func/*brain_mask.* \
            contarg/test/data/derivatives/fmriprep/sub-*/func/*boldref.* \
            contarg/test/data/derivatives/fmriprep/sub-*/func/*confounds_timeseries.* \
            contarg/test/data/derivatives/contarg/hierarchical/testing*_ref \
            contarg/test/data/derivatives/contarg/hierarchical/testing1sub/**/*mask.nii.gz \
            contarg/test/data/derivatives/contarg/hierarchical/testing2subs/**/*mask.nii.gz 
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

## Implemented methods
* Hierarchical clustering
    *   `contarg hierarchical run`
* Seedmap + classic
    *   `contarg seedmap run --targeting-method=classic`
* Seedmap + cluster
    *   `contarg seedmap run --targeting-method=cluster`
* TANS + refernce correlation
    *   `contarg tans run --target-method=reference`
* TANS + depression circuit spatial similarity 
    *   `contarg tans run --target-method=DCSS`

For Seedmap methods you should provide a seedmap in MIN152NLin6Asym space.
You can use the one in test/data/derivatives/contarg/seedmap/hcp_working, 
remember you may need to download it with `datalad get`. 
Alternatively, you can use `contarg seedmap subjectmap` and `contarg seedmap groupmap` to make one.

TANS is Targeted Functional Network Stimulation implemented by Chuck Lynch. I've just wrapped it in python and made a 
CLI. In order to run that, you will need Matlab, Freesurfer, FSL, connectome-workbench, ANTs, and SimNibs 4.0.

## ROIs
We've vot 4 reference masks and 4 stim masks. Most were created via the AFNI commands below. coleBA46 was based on extraction of MNI 
coordinates from Cole et al., 2020. We hand traced an ROI that enclosed those points and followed gyral landmarks on 
MNI152NLin6Asym (1mm resolution). We then cleaned up the hand traced mask via inlation/delation and fit it with a convex
hull using Mango. 
```commandline
# make SGC mask 10 mm sphere at MNI 6,16,-10
3dcalc -a ~/.cache/templateflow/tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-02_T1w.nii.gz   \
       -expr 'step(100-(x+6)*(x+6)-(y+16)*(y+16)-(z+10)*(z+10))' \
       -prefix SGCsphere_space-MNI152NLin6Asym_res-02.nii.gz
# make blSGC mask 10 mm spheres at (-6,16,-10 and 6,16,-10)
3dcalc -a ~/.cache/templateflow/tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-02_T1w.nii.gz   \
       -expr 'step(100-(x-6)*(x-6)-(y+16)*(y+16)-(z+10)*(z+10))' \
       -prefix leftSGCsphere_space-MNI152NLin6Asym_res-02.nii.gz
3dcalc -a SGCsphere_space-MNI152NLin6Asym_res-02.nii.gz   \
       -b leftSGCsphere_space-MNI152NLin6Asym_res-02.nii.gz    \
       -expr 'step(a+b)' \
       -prefix bilateralSGCspheres_space-MNI152NLin6Asym_res-02.nii.gz
# make a bilateral full SGC mask 12 mm sphere at 0, 24, -14
3dcalc -a ~/.cache/templateflow/tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-02_T1w.nii.gz   \
       -expr 'step(144-(x-0)*(x-0)-(y+24)*(y+24)-(z+14)*(z+14))' \
       -prefix bilateralfullSGCsphere_space-MNI152NLin6Asym_res-02.nii.gz
# BA9 20mm -36, 39, 43
3dcalc -a ~/.cache/templateflow/tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-02_T1w.nii.gz   \
       -expr 'step(400-(x-36)*(x-36)-(y+39)*(y+39)-(z-43)*(z-43))' \
       -prefix BA9sphere_space-MNI152NLin6Asym_res-02.nii.gz

# BA46 20mm -44, 40, 29
3dcalc -a ~/.cache/templateflow/tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-02_T1w.nii.gz   \
       -expr 'step(400-(x-44)*(x-44)-(y+40)*(y+40)-(z-29)*(z-29))' \
       -prefix BA46sphere_space-MNI152NLin6Asym_res-02.nii.gz

# 5cm 20mm -41, 16, 54
3dcalc -a ~/.cache/templateflow/tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-02_T1w.nii.gz   \
       -expr 'step(400-(x-41)*(x-41)-(y+16)*(y+16)-(z-54)*(z-54))' \
       -prefix 5cmsphere_space-MNI152NLin6Asym_res-02.nii.gz

# F3 20mm -37, 26, 49
3dcalc -a ~/.cache/templateflow/tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-02_T1w.nii.gz   \
       -expr 'step(400-(x-37)*(x-37)-(y+26)*(y+26)-(z-49)*(z-49))' \
       -prefix F3sphere_space-MNI152NLin6Asym_res-02.nii.gz
       
# BA9 30mm -36, 39, 43
3dcalc -a ~/.cache/templateflow/tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-02_T1w.nii.gz   \
       -expr 'step(900-(x-36)*(x-36)-(y+39)*(y+39)-(z-43)*(z-43))' \
       -prefix dilatedBA9sphere_space-MNI152NLin6Asym_res-02.nii.gz

# BA46 30mm -44, 40, 29
3dcalc -a ~/.cache/templateflow/tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-02_T1w.nii.gz   \
       -expr 'step(900-(x-44)*(x-44)-(y+40)*(y+40)-(z-29)*(z-29))' \
       -prefix dilatedBA46sphere_space-MNI152NLin6Asym_res-02.nii.gz

# 5cm 30mm -41, 16, 54
3dcalc -a ~/.cache/templateflow/tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-02_T1w.nii.gz   \
       -expr 'step(900-(x-41)*(x-41)-(y+16)*(y+16)-(z-54)*(z-54))' \
       -prefix dilated5cmsphere_space-MNI152NLin6Asym_res-02.nii.gz

# F3 30mm -37, 26, 49
3dcalc -a ~/.cache/templateflow/tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-02_T1w.nii.gz   \
       -expr 'step(900-(x-37)*(x-37)-(y+26)*(y+26)-(z-49)*(z-49))' \
       -prefix dilatedF3sphere_space-MNI152NLin6Asym_res-02.nii.gz

# DLPFC
3dcalc -a BA9sphere_space-MNI152NLin6Asym_res-02.nii.gz -b BA46sphere_space-MNI152NLin6Asym_res-02.nii.gz \
       -c 5cmsphere_space-MNI152NLin6Asym_res-02.nii.gz -d F3sphere_space-MNI152NLin6Asym_res-02.nii.gz  \
       -expr 'step(a + b + c + d)'  \
       -prefix DLPFCspheres_space-MNI152NLin6Asym_res-02.nii.gz
       
# Dilated DLPFC
3dcalc -a dilatedBA9sphere_space-MNI152NLin6Asym_res-02.nii.gz -b dilatedBA46sphere_space-MNI152NLin6Asym_res-02.nii.gz \
       -c dilated5cmsphere_space-MNI152NLin6Asym_res-02.nii.gz -d dilatedF3sphere_space-MNI152NLin6Asym_res-02.nii.gz  \
       -expr 'step(a + b + c + d)'  \
       -prefix dilatedDLPFCspheres_space-MNI152NLin6Asym_res-02.nii.gz

# DLPFC + brainmask
3dcalc -a BA9sphere_space-MNI152NLin6Asym_res-02.nii.gz -b BA46sphere_space-MNI152NLin6Asym_res-02.nii.gz \
       -c 5cmsphere_space-MNI152NLin6Asym_res-02.nii.gz -d F3sphere_space-MNI152NLin6Asym_res-02.nii.gz  \
       -e ~/.cache/templateflow/tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-02_desc-brain_T1w.nii.gz \
       -expr 'and(step(a + b + c + d),e)'  \
       -prefix DLPFCspheresmasked_space-MNI152NLin6Asym_res-02.nii.gz
       
# BA46 + brainmask
3dcalc -a /usr/local/apps/fsl/6.0.4/data/standard/MNI152_T1_2mm.nii.gz   \
       -b ~/.cache/templateflow/tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-02_desc-brain_T1w.nii.gz \
       -expr 'and(a,b)' \
       -prefix BA46spheremasked_space-MNI152NLin6Asym_res-02.nii.gz
```


This is a Python project packaged according to [Contemporary Python Packaging - 2023][].

[Contemporary Python Packaging - 2023]: https://effigies.gitlab.io/posts/python-packaging-2023/
