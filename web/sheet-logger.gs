/**
 * sheet-logger.gs
 *
 * Paste this into Extensions > Apps Script on a Google Sheet, then
 * Deploy > New deployment > Web app (execute as Me, access: Anyone).
 * Copy the resulting URL into SHEET_ENDPOINT in spectre-tool.html.
 *
 * Appends one row per Spectre submission to a sheet named "Submissions"
 * (created automatically on first request if it doesn't exist).
 */
function doPost(e) {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var sheet = ss.getSheetByName('Submissions');
  if (!sheet) {
    sheet = ss.insertSheet('Submissions');
    sheet.appendRow([
      'Timestamp', 'Name', 'Institute', 'Spectrum Source',
      'Peaks Detected', 'Top Functional Group', 'Top Confidence (%)'
    ]);
  }

  var data = JSON.parse(e.postData.contents);

  sheet.appendRow([
    new Date(data.timestamp),
    data.name || '',
    data.institute || '',
    data.source || '',
    data.peakCount || '',
    data.topGroup || '',
    data.topConfidence || ''
  ]);

  return ContentService
    .createTextOutput(JSON.stringify({ status: 'ok' }))
    .setMimeType(ContentService.MimeType.JSON);
}

/**
 * Optional: lets you sanity-check the deployment URL by visiting it
 * directly in a browser (GET requests only, does not write any rows).
 */
function doGet(e) {
  return ContentService
    .createTextOutput(JSON.stringify({ status: 'Spectre logger is live' }))
    .setMimeType(ContentService.MimeType.JSON);
}
