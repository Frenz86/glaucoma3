import streamlit as st
from pag1 import main as  pag1
from pag2 import main as  pag2
#from pagxx import main as pagxx

#to monitor logging
import csv
from datetime import datetime as dt
import base64
import gspread

def get_time():
    d = dt.now().strftime("%Y-%m-%d-%H:%M:%S")
    return d

def write_data_on_csv(filename, listdata):
    with open(filename, mode='a') as f:
        fw = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        fw.writerow(listdata)

name = 'PROJECT-XXXXX'  # name of app on gspread
## credential.json is the credential file for google sheet permission
#######################################################

# Security
#passlib,hashlib,bcrypt,scrypt
import hashlib
def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False
# DB Management
import sqlite3 
conn = sqlite3.connect('data.db')
c = conn.cursor()
# DB  Functions
def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data

def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data


def main():
	# ################ css background #########################
	page_bg_img = '''
	<style>
	body {
	background-image: url("https://i.pinimg.com/originals/85/6f/31/856f31d9f475501c7552c97dbe727319.jpg");
	background-size: cover;
	}
	</style>
	'''
	st.markdown(page_bg_img, unsafe_allow_html=True)

	################ load logo from web #########################
	from PIL import Image
	import requests
	from io import BytesIO
	#url='https://frenzy86.s3.eu-west-2.amazonaws.com/python/sfondo.png'
	#response = requests.get(url)
	#image = Image.open(BytesIO(response.content))
	st.title("AI APP to predict glaucoma through fundus image of eye")
	image = Image.open('the-biggest.jpg')
	st.image(image, caption='',use_column_width=True)
	##############################################################
	#menu = ["Login","SignUp"] # per creare password
	menu = ["Login"]
	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "Login":
		st.subheader("")

		username = st.sidebar.text_input("User Name")
		password = st.sidebar.text_input("Password",type='password')
		if st.sidebar.checkbox("Login"):
			# if password == '12345':
			create_usertable()
			hashed_pswd = make_hashes(password)

			result = login_user(username,check_hashes(password,hashed_pswd))
			if result:
				#st.success("Logged In as {}".format(username))
				######## monitor access ########
								#st.success("Logged In as {}".format(username))
				log = ("{}".format(username))
				time_print = get_time()
				write_data_on_csv(filename="login.csv",listdata=[time_print,name,log]) # to wite csv in local
				print("Logged In as {}".format(username))
				#gc = gspread.service_account(filename='credentials.json')
				#sh = gc.open("Login_webapp")
				#worksheet = sh.sheet1
				#listdata=[time_print,name,log]
				#worksheet.append_row(listdata)# no specify the row

				#### PAGES SETTINGS   ##############
				#pag_name = ["pag1","pag2","pag3","pag4","pag5"]
				pag_name = ["Prediction Model",'Instruction to crop the images']
				
				OPTIONS = pag_name
				sim_selection = st.selectbox('Seleziona la pagina', OPTIONS)

				if sim_selection == pag_name[0]:
					pag1()
				elif sim_selection == pag_name[1]:
					pag2()
				elif sim_selection == pag_name[2]:
					pag3()
				elif sim_selection == pag_name[3]:
					pag4()
				elif sim_selection == pag_name[4]:
					pag5()
				else:
					st.markdown("Something went wrong. We are looking into it.")

			else:
				st.warning("Incorrect Username/Password")
	else:
		st.subheader("Create New Account")
		new_user = st.text_input("Username")
		new_password = st.text_input("Password",type='password')

		if st.button("Signup"):
			create_usertable()
			add_userdata(new_user,make_hashes(new_password))
			st.success("You have successfully created a valid Account")
			st.info("Go to Login Menu to login")

if __name__ == '__main__':
	main()