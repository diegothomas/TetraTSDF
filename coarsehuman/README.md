# Coarse human outer shell

To generate TSDFs and network connections in PCN layers, we need to prepare coarse human outer shell and voxelize inside of it.

This code contains already processed data, but if you want to use another coarse human, follow the instruction below.

## Requirements
- python2.7

- chumpy

- SMPL and SMPLify setup  
Download the SMPL and SMPLify code from their official website and place smpl_webuser and smplify_public to ./coarsehuman  
[SMPL](https://smpl.is.tue.mpg.de/)  
[SMPLify](http://smplify.is.tue.mpg.de/)  

## Procedure
1. Prepare your own coarse human by using some tools like Blender.


2. Run tetrahedralize.py and createweights.py on your python2.7 environment  

```
cd [TetraTSDF root]/coarsehuman
python tetrahedralize.py --plypath [path to coarse human] --create_adjlists
python createweights.py --starpose
```
You need male/female/neutral SMPL model to run this code.  

Download SMPL models from official website and place  
basicModel_f_lbs_10_207_0_v1.0.0.pkl  
basicModel_m_lbs_10_207_0_v1.0.0.pkl  
basicModel_neutral_lbs_10_207_0_v1.0.0.pkl  
in [TetraTSDF root]/coarsehuman/models folder.


3. Place resulted .csv files  
adjlist_1to0.csv  
adjlist_2to1.csv  
adjlist_3to2.csv  
adjlist_4to3.csv  
to [TetraTSDF root]/network/adjLists