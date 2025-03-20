import socket,gspread,datetime,time, platform, requests
import pandas as pd

class GoogleController:
	def __init__(self, credentialSpreadsheet, nonPiFlag = False):
		self.credentialSpreadsheet = credentialSpreadsheet
		self.nonPiFlag = nonPiFlag

		self._authenticateGoogleSpreadSheets() #Creates self.controllerGS
		if not self.nonPiFlag:
			self._identifyTank() #Stored in self.tankID
			self._identifyServiceAccount()

	def addProjectID(self, projectID, googleErrorFile):
		self.projectID = projectID
		self.googleErrorFile = googleErrorFile
		self.g_lf = open(self.googleErrorFile, 'a', buffering = 1)

	def getPiGS(self, column_names):
		# Make this compatible with both lists and also strings
		if not isinstance(column_names, list):
			column_names = [column_names]
		for i in range(3):
			try:
				#print('Read request: ' + str(datetime.datetime.now()))
				data = self.pi_ws.get_all_values()
			except gspread.exceptions.APIError as e:
				if e.response.status_code == 429:
				# Read requests per minute exceeded
					self._googlePrint('Read requests per minute exceeded')
					continue
				elif e.response.status_code == 500:
					self._googlePrint('Internal error encountered')
					continue
				elif e.response.status_code == 404:
					self._googlePrint('Requested entity was not found')
					continue
				else:
					self._googlePrint('gspread error of unknown nature: ' + str(e))
					continue
			except requests.exceptions.ReadTimeout as e:
				self._googlePrint('Requests read timeout error encountered')
				continue
			except Exception as e:
				self._googlePrint(f'uncaught exception in _getPiGS: {str(e)}')
				continue

			except requests.exceptions.ConnectionError as e:
				self._googlePrint('Requests connection error encountered')
				continue


			dt = pd.DataFrame(data[1:], columns = data[0])
			self.dt = dt
			out_data = []
			for column_name in column_names:
				if column_name not in dt.columns:
					self._googlePrint('Cant find column name in Controller: ' + column_name)
					raise Exception
				try:
					cell = dt.loc[(dt.RaspberryPiID == platform.node())&(dt.IP == self.IP),column_name]
				except AttributeError as error:
					pdb.set_trace()
				if len(cell) > 1:
					self._googlePrint('Multiple rows in the Controller with the same ID and IP. Using 1st')
					self.modifyPiGS('Error', 'InstructError: Multiple rows with the same IP/ID', ping = False)
				out_data.append(cell.values[0])

			if len(out_data) == 1:
				return out_data[0]
			else:
				return out_data
		self._googlePrint('Failed contancting controller for three tries')
		return [None]*len(column_names)

	def modifyPiGS(self, column_name, new_value, ping = True):
		for i in range(3):
			try:
				row, column, ping_column = self._getRowColumn(column_name)

				#print('Write request: ' + str(datetime.datetime.now()))
				self.pi_ws.update_cell(row, column, new_value)

				if ping:
					#print('Write request: ' + str(datetime.datetime.now()))
					self.pi_ws.update_cell(row, ping_column, str(datetime.datetime.now()))
				break
			except gspread.exceptions.APIError as e:
				if e.response.status_code == 429:
					# Read requests per minute exceeded
					self._googlePrint('Read requests per minute exceeded')
					continue
				elif e.response.status_code == 500:
					self._googlePrint('Internal error encountered')
					continue
				elif e.response.status_code == 404:
					self._googlePrint('Requested entity was not found')
					continue
				else:
					self._googlePrint('gspread error of unknown nature: ' + str(e))
					continue
			except requests.exceptions.ReadTimeout:
				self._googlePrint('Read timeout error')
				continue

	def _authenticateGoogleSpreadSheets(self):

		# Get IP address
		s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		s.connect(("8.8.8.8", 80))
		self.IP = s.getsockname()[0]
		s.close()

		# Try to authenticate three times before returning error
		for i in range(0,3): # Try to autheticate three times before failing
			try:
				# gs = gspread.authorize(credentials)
				gs = gspread.service_account(filename=self.credentialSpreadsheet)
			except Exception as e:
				self._googlePrint(e)
				continue
			try:
				self.controllerGS = gs.open('Controller')
				self.pi_ws = self.controllerGS.worksheet('RaspberryPi')
				data = self.pi_ws.get_all_values()
				dt = pd.DataFrame(data[1:], columns = data[0])
			except Exception as e:
				self._googlePrint(e)
				continue
			if not self.nonPiFlag:
				try:
					if len(dt.loc[dt.RaspberryPiID == platform.node()]) == 0:
						self.pi_ws.append_row([platform.node(),self.IP,'','','','','','None','Stopped','Error: Awaiting assignment of TankID',str(datetime.datetime.now())])
						return True
					else:
						return True
				except Exception as e:
					self._googlePrint(e)
					continue
			else:
				self.all_data = data
				time.sleep(2)
		return False

	def _identifyTank(self):
		while True:
			tankID = self.getPiGS('TankID')
			if tankID not in ['None','']:
				self.tankID = tankID
				break
			else:
				self.modifyPiGS('Error','Awaiting assignment of TankID')
				time.sleep(20)

	def _identifyServiceAccount(self):
		while True:
			serviceAccount = self.getPiGS('ServiceAccount')
			if serviceAccount not in ['None','']:
				self.serviceAccount = serviceAccount
				self.credentialSpreadsheet = self.credentialSpreadsheet.replace('_1.json', '_' + self.serviceAccount + '.json')
				self._authenticateGoogleSpreadSheets() #Creates self.controllerGS

				break
			else:
				self.modifyPiGS('Error','Awaiting assignment of ServiceAccount')
				time.sleep(20)

	def _googlePrint(self, e):
		try:
			print(str(datetime.datetime.now()) + ': ' + str(type(e)) + ': ' + str(e), file = self.g_lf, flush = True)
			time.sleep(20)
		except AttributeError as e2: # log file not created yet so just print to stderr
			print(str(datetime.datetime.now()) + ': ' + str(type(e)) + ': ' + str(e), flush = True)
			time.sleep(20)

	def _getRowColumn(self, column_name):
		column = self.dt.columns.get_loc(column_name)
		ping_column = self.dt.columns.get_loc('Ping')
		row = pd.Index((self.dt.RaspberryPiID == platform.node())&(self.dt.IP == self.IP)).get_loc(True)
		return (row + 2, column + 1, ping_column + 1) # 0 vs 1 indexing for pandas vs gspread + column names aren't in the pandas dataframe