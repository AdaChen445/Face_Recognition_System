# -*- coding: utf-8 -*-

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import datetime
import ssl

class PythonEmail():
	def __init__(self):
		self.gmail_user = 'B10630037@gapps.ntust.edu.tw'
		self.gmail_password = '##############'
		self.from_address = 'B10630037@gapps.ntust.edu.tw'
		self.succ_subject = "通知:您的人臉辨識系統註冊成功"
		self.succ_contents = """
		您好，
			感謝您使用台科大多媒體通訊實驗室專題生寫的人臉辨識點名系統
			您的人臉樣本已成功註冊，期待您親自使用本系統!

		祝 平安喜樂、期末歐趴準時畢業、吃飽睡飽心情舒暢、年薪300萬財富自由
		"""
		self.fail_subject = "通知:您的人臉辨識系統註冊失敗"
		self.fail_contents = """
		您好，
			很遺憾您的人臉辨識系統註冊失敗，請重新錄製影片後至註冊連結再次上傳
			請注意錄製過程中請勿戴帽子、勿遮蔽臉部及眉毛，並於錄製完畢後將影片重新命名為 ID_Email

		祝 平安喜樂、期末歐趴準時畢業、吃飽睡飽心情舒暢、年薪300萬財富自由
		"""


	def sendmail(self, condition, address):
		to_address = str(address)
 
		mail = MIMEMultipart()
		mail['From'] = self.from_address
		mail['To'] = to_address
		if condition:
			mail['Subject'] = self.succ_subject
			mail.attach(MIMEText(self.succ_contents))
		else:
			mail['Subject'] = self.fail_subject
			mail.attach(MIMEText(self.fail_contents))

		smtpserver = smtplib.SMTP("smtp.gmail.com", 587)
		smtpserver.ehlo()
		smtpserver.starttls()
		smtpserver.login(self.gmail_user, self.gmail_password)
		status = smtpserver.sendmail(self.from_address, to_address, mail.as_string())
		if status == {}:
			print("[INFO][EM] EMAIL SEND SUCCESS")
		else:
			print("[INFO][EM] EMAIL SEND FAIL")
		smtpserver.quit()