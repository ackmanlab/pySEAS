### Copying the repo into local computer 

I copied the repo into a local computer repo and from Sydney's branch, I made a branch called desi 


### Installing the dependencies 

I opened the readme file and I followed the instructions to install the necessary dependencies. When I put in the code 

pip install -r requirements.txt

I was given a depreciation message about python3.5. I am first doing this round of everything using python 3.5. I'll then update to the most recent version (3.8). 

When I executed the pip install -r requirements.txt code I got two errors: 

ERROR: Could not find a version that satisfies the requirement boto3==1.17.13
ERROR: No matching distribution found for boto3==1.17.13

I just erased the boto3 dependencies from the text file 


I also got this error:

ERROR: Could not find a version that satisfies the requirement h5==0.4.1
ERROR: No matching distribution found for h5==0.4.


Going to just try to run the example functions without completely installing all the dependencies


### Running the example functions 


play_dfof_movie.py 

Worked well with both tiff files, if I was new new to this I would want to see from the downsample option what the expected input type is (integer) and then also what the range is, x-y numbers 

project_ica.py 

It was intuitive how to use the set up the -f and -e flags, for the folder and experiment respectively. I  first tried to run it without using the -rt flag but it did not work, I saw there was a rotation comparison errror going on and assumed if I rotated it by 1 x 90-degrees it might work and it did. 



