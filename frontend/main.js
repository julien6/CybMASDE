const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const isDev = !app.isPackaged; // Détecte si l'application est en mode dev ou prod


// Recharge automatique si en développement
if (isDev) {
  require('electron-reload')(path.join(__dirname, 'dist/frontend'), {
    electron: path.join(__dirname, 'node_modules', '.bin', 'electron'),
    awaitWriteFinish: true, // Petit délai pour éviter les rechargements rapides
  });
}

function createWindow() {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'), // Spécifie le fichier preload ici
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  // Masquer la barre de menu
  win.setMenuBarVisibility(false);

  // Charger la page d'accueil
  // win.loadURL(`file://${path.join(__dirname, 'dist/frontend/browser/index.html')}`);
  win.webContents.openDevTools();

  // Charge l'URL de développement ou les fichiers locaux en fonction du mode
  if (isDev) {
    win.loadURL('http://localhost:4200'); // Utilise `ng serve` en développement
  } else {
    win.loadFile(path.join(__dirname, '/dist/frontend/browser/index.html'));
  }

}

app.whenReady().then(() => {
  // Démarrage du backend Python en activant l'environnement virtuel
  const pythonProcess = spawn('bash', [
    '-c',
    `source ${path.join(__dirname, '../backend/venv/bin/activate')} && python ${path.join(__dirname, '../backend/src/api_server/server.py')}`
  ]);

  ipcMain.on('open-save-dialog', async (event) => {
    const result = await dialog.showSaveDialog({
      title: 'Enregistrer le fichier',
      defaultPath: 'myFile.txt', // Optionnel: nom par défaut
      filters: [
        { name: 'Text Files', extensions: ['txt'] }, // Vous pouvez personnaliser selon votre besoin
        { name: 'All Files', extensions: ['*'] },
      ],
    });

    // Vérifie si un chemin a été sélectionné et renvoie le chemin
    if (!result.canceled && result.filePath) {
      event.sender.send('save-file-response', result.filePath);
    } else {
      event.sender.send('save-file-response', null); // Si l'utilisateur annule
    }
  });

  pythonProcess.stdout.on('data', (data) => {
    console.log(`Python server: ${data}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`Python server error: ${data}`);
  });

  app.on('before-quit', () => {
    pythonProcess.kill(); // Fermer le serveur Python proprement
  });

  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });

  // Écouter l'événement "close-app" depuis Angular
  ipcMain.on('close-app', () => {
    app.quit(); // Termine l'application
  });

});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
