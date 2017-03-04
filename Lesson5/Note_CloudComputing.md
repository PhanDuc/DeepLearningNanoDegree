# Cloud Computing

## Environment

- AMAZON AWS EC2: [LINK](https://us-west-1.console.aws.amazon.com/console/home?region=us-west-1) 
    - Register New Account [Checked] 
        - Have used my Gmail for log-in
    - Apply and Retrieve Credits [Check]
    - Launch an instance [Waiting]
    - Tutorials: [Link](https://aws.amazon.com/getting-started/?sc_channel=em&sc_campaign=wlcm&sc_publisher=aws&sc_medium=em_wlcm_2&sc_detail=wlcm_2b&sc_content=other&sc_country=global&sc_geo=global&sc_category=mult&ref_=pe_1679150_132208650)

- AWS Training: [Link](https://awstraining.csod.com/)
    - User Name: Emails for udacity
    - Password: 
        - Passwords must contain both upper and lower case letters
        - Passwords must contain alpha and numeric characters
        - Passwords cannot be the same as the previous 3 passwords
        - Passwords must be 6 - 20 characters   
        - Passwords cannot have leading or trailing spaces
        - Passwords cannot be the same as the Username, User ID, or email address

- General AWS education credit
    - Application [Link](https://www.awseducate.com/InstitutionApplication) 

## Setup the GPU

1. Choose the instance
    - Elastic Compute Cloud (EC2) -- Use `g2.2xlarge` Instance
2. Check the service limit report and Submit a Limit Increase Request

## Login AWS

1. Get IPv4 addresion: 
    - Note the "IPv4 Public IP" address (in the format of “X.X.X.X”) on the EC2 Dashboard.
2. Go to your terminal
3. Type `ssh udacity@X.X.X.X`
4. Authenticate with the password "udacity".

## Testing

### On the EC2 instance

1. Clone a repo of TensorFlow Examples: `git clone https://github.com/udacity/deep-learning.git`
2. Enter the repo directory `cd deep-learning/intro-to-tensorflow/`
3. Activate the new environment: `source activate dl`
4. Run the notebook: `jupyter notebook`

### From your local machine

1. Access the Jupyter notebook index from your web browser by visiting: X.X.X.X:8888 (where X.X.X.X is the IP address of your EC2 instance)
2. Click on the "intro_to_tensorflow.ipynb" link to launch the LeNet Lab Solution notebook
3. Run each cell in the notebook
