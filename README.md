# Salary Predictions Based on Job Descriptions

I split this project into the 4 main processes in a data science project: [Defining the Problem](01_Define.ipynb), [Exploring 
the Data](02_Discover.ipynb), [Developing the Model](03_Develop.ipynb), and [Deploying the Model](04_Deploy.ipynb). The Define
notebook goes through the research to gain domain knowledge on the problem as well as assesses what kind of information is 
available in the data. The Discover notebook holds all the EDA as well as introduces some of the functions that will be useful 
for analysis such as load_data and find_outliers. The Developing notebook creates and tunes various models to see which would 
be the best for this appication based on minimizing the error chosen for this project. The Deploying notebook actually creates
the pipeline and deploys the model. Since this is relatively short, I also include an analysis here on how I want to improve 
going forward, both on this project as well as projects in the future. 

Found in the [scripts](scripts/) folder are the files used to help me throughout the project. NaiveModel.py provides the 
baseline model. Preprocessing.py holds the preprocessing object used in developing the pipeline. helpers.py has all the 
functions that were utilized throughout the notebooks and model building.

If the jupyter notebooks fail to load, there are markdown backups of every file in the [backups](backups/) folder. These should
work whether or not GitHub can currently handle Jupyter NoteBooks.
