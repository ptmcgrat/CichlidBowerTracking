/*
  10/19/24 JKS
  Script: ControllerSheetAppsScript.js
  Purpose: Google Apps Script that will parse the Controller sheet for errors, then send an email.

  Set up ranges on Google Sheet:
	1. Go to Data -> Named ranges
	2. Add the following ranges:
	   a) RaspberryPiID: RaspberryPi!A2:A9
	   b) Command: RaspberryPi!I2:I9
	   c) Status: RaspberryPi!J2:J9
	   d) Error: RaspberryPi!K2:K9
	   e) Ping: RaspberryPi!L2:L9
	   f) DataDuplicated: RaspberryPi!O2:O9
	
  Set up Apps Script:
    1. Open the Controller Sheet ()
    2. Navigate to Extensions -> Apps Script
    3. Copy and paste this script into the project
    4. Under Services, add "GoogleSheets API."
    5. Under Triggers, Add a Time-Based Trigger with the following options selected:
       a) Choose which function to run: validateCommand
       b) Choose which deployment to run: Head
       c) Select event source: From spreadsheet
       d) Select event type: On open
       e) Failure notification settings: Notify me immediately

  Don't forget:
    1. Add all email recipeints to emails.
	2. Add any Pis you don't want to receive errors to excludePis. 
*/

const emails = ['caitmsullivan7@gmail.com', 'jeanettek.schofield@gmail.com']; // Add email to list to receive Controller Sheet Errors
const excludePis = ['bt-t033', 'bt-t047']; // Skip these Pis when checking for errors
const debug = false

function validateCommand(command) {
  return command && (String(command).toLowerCase() == 'none')
}

function validateStatus(status) {
  return status && (String(status).toLowerCase() == 'running')
}

function validateError(error) {
  return !error
}

function validatePing(ping) {
  var now = new Date();
  const delta = Math.abs(now - ping)/60000

  return (delta < 15)
}

function validateDataDuplicated(dataDuplicated) {
  return dataDuplicated && (String(dataDuplicated).toLowerCase() == 'no')
}

function sendEmail(recipient, pis, errors) {
  const subject = 'Controller Sheet Errors, PIs: ' + pis
  var htmlBody ='<H3>Errors Found:</H3>' 
  
  for (i=0; i<errors.length; i++) {
    htmlBody += errors[i] + '<br>'
  }

  MailApp.sendEmail({
    to: recipient, 
    subject: subject, 
    htmlBody: htmlBody
  })
}

function checkForErrors() {
  
  const errors = [];
  const pis = [];

  try {
    const col_RaspberryPiIDs = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet().getRange('RaspberryPiID').getValues()
    const col_Command = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet().getRange('Command').getValues()
    const col_Status = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet().getRange('Status').getValues()
    const col_Error = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet().getRange('Error').getValues()
    const col_Ping = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet().getRange('Ping').getValues()
    const col_DataDuplicated = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet().getRange('DataDuplicated').getValues()

    for (let i = 0; i < col_RaspberryPiIDs.length; i++)
    {
      const curr_RaspberryPiID = col_RaspberryPiIDs[i][0]
      if (!curr_RaspberryPiID || excludePis.includes(curr_RaspberryPiID)) { continue }
      
      const curr_Command = col_Command[i][0]
      const curr_Status = col_Status[i][0]
      const curr_Error = col_Error[i][0]
      const curr_Ping = col_Ping[i][0]
      const curr_DataDuplicated = col_DataDuplicated[i][0]

      // Add Error for Command column
      if (!validateCommand(curr_Command)) {
        pis.push(curr_RaspberryPiID)
        errors.push('Pi ' + curr_RaspberryPiID + ': Command = ' + curr_Command)
      }

      // Add Error for Status column
      if (!validateStatus(curr_Status)) {
        if (!pis.includes(curr_RaspberryPiID)) { pis.push(curr_RaspberryPiID) }
        errors.push('Pi ' + curr_RaspberryPiID + ': Status = ' + curr_Status)
      }

      // Add Error for Error column
      if (!validateError(curr_Error)) {
        if (!pis.includes(curr_RaspberryPiID)) { pis.push(curr_RaspberryPiID) }
        errors.push('Pi ' + curr_RaspberryPiID + ': Error = ' + curr_Error)
      }

      // Add Error for Ping column
      if (!validatePing(curr_Ping)) {
        if (!pis.includes(curr_RaspberryPiID)) { pis.push(curr_RaspberryPiID) }
        errors.push('Pi ' + curr_RaspberryPiID + ': Ping > 15 mins (Last ping: ' + String(curr_Ping) + ')')
      }

      // Add Error for DataDuplicated column
      if (!validateDataDuplicated(curr_DataDuplicated)) {
        if (!pis.includes(curr_RaspberryPiID)) { pis.push(curr_RaspberryPiID) }
        errors.push('Pi ' + curr_RaspberryPiID + ': DataDuplicated = ' + curr_DataDuplicated)
      }
    }

    // Print errors if debug = true
    if (debug) {
      for (let i = 0; i < errors.length; i++) {
        console.log(errors[i])
      }

      console.log('Pis with Errors: ' + pis)
    }

    // Send email
    if (errors.length > 0)
    {
      for (let i = 0; i < emails.length; i++) {
        try {          
          sendEmail(emails[i], pis, errors);
        } catch (err) {
          console.log(error);
        }
      }
    }

  } catch (err) {
    console.log(err.message);
  }
}
