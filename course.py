# This is the model for which course data is stored in.
# Since the data was in near identical layouts, you'll
# notice very specific queries were use to grab the data
# on some pieces.

from bs4 import BeautifulSoup

def getComments(id, soup):
	valuable = soup.find("ul", {"id": id}).text.split("\n")
	realValuable = []
	for comment in valuable:
		comment = comment.lstrip()
		if not len(comment) == 0:
			realValuable += [comment]

	return realValuable

class Course():
	#code: 			Department & Number
	#semester		Semester & Year
	#courseType		LEC/LAB/SEM...
	#professors		String professor names
	#courseName		String course name
	#studentNum		int enrolled students
	#avgGrade		float average grade
	#gradesVector	int array of earned grades (A, A-, ..., P, NC, I, W)
	#valuableComm	string array of 'what was valuable' comments
	#improvedComm	string array of 'what could be improved' comments
	#attendance		int array of reported attendance stats
	#studying		int array of reported studying stats
	def __init__(self,fileName):
		#Create BS
		with open(fileName, 'rb') as file:
			soup = BeautifulSoup(file, 'html.parser')

		#Parse title
		maybeTitle = soup.findAll("h1", {"class": "centered animated fadeInDown white"})[0].text.split('\n')
		titlePieces = []
		for title in maybeTitle:
			title = title.lstrip()
			if len(title) > 0:
				titlePieces += [title]

		self.code = titlePieces[0][0:6]
		self.semester = titlePieces[1]
		self.courseType = titlePieces[2]

		#Additionally split by ' and ' & ', '
		professors = soup.find("div", {"class": "col-sx-12 col-sm-12 col-md-12"}).findChildren('h3')[0].text
		self.professors = professors

		courseName = soup.find("div", {"class": "col-sx-12 col-sm-12 col-md-12"}).findChildren('h2')[0].text
		self.courseName = courseName

		scriptTag = soup.findAll('script')[7].text.split('\n')
		totalUsers = 0
		gradeConstant = 4.0
		avgGrade = 0
		gradesVector = []
		#for weird cases
		rangeVal = 20
		#Relevant script lines, always
		if scriptTag[4].replace(r'\s', '') == '\r':
			rangeVal = 16
		for i in range(rangeVal, rangeVal + 15):
			numberGiven = int(scriptTag[i].split(',')[1])
			gradesVector += [numberGiven]
			if not gradeConstant < 0:
				avgGrade += gradeConstant * numberGiven
				gradeConstant -= .3
			totalUsers += numberGiven

		self.studentNum = totalUsers
		
		#Should only execute if people were in the course
		if not totalUsers == 0:
			self.avgGrade = avgGrade/totalUsers
			self.gradesVector = gradesVector
		else:
			self.avgGrade = 0
			self.gradesVector = [0] * 15

		try:
			valuable = getComments("paginate-1", soup)
		except:
			valuable = []
		try:
			improved = getComments("paginate-2", soup)
		except:
			improved = []

		self.valuableComm = valuable
		self.improvedComm = improved


		miscMetrics = soup.findAll('div', {'class': 'col-sm-12 col-lg-6'})
		if len(miscMetrics) > 0:
			studying = miscMetrics[0].text.split('\n')
			attendance = miscMetrics[1].text.split('\n')
			self.attendance = [float(attendance[3]), float(attendance[5]), float(attendance[7])]
			self.studying = [float(studying[3]), float(studying[5]), float(studying[7])]
		else:
			self.attendance = []
			self.studying = []

			