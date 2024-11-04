const { app, BrowserWindow, ipcMain, dialog, globalShortcut, shell } = require('electron');
const { spawn } = require('child_process');
const { format } = require('url');
const path = require('path');


const isDev = !app.isPackaged; // Détecte si l'application est en mode dev ou prod

// Recharge automatique en mode développement
if (isDev) {
  require('electron-reload')(path.join(__dirname, 'dist/frontend'), {
    electron: path.join(__dirname, 'node_modules', '.bin', 'electron'),
    awaitWriteFinish: true,
  });
}

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
      // enableRemoteModule: false,
      // sandbox: false
    },
  });

  mainWindow.setMenuBarVisibility(true); // Masquer la barre de menu

  // Charge l'URL de développement ou le fichier local selon le mode
  if (isDev) {
    mainWindow.loadURL('http://localhost:4200');
    mainWindow.webContents.openDevTools(); // Ouvre DevTools en mode dev

  } else {
    // mainWindow.loadFile(path.join(__dirname, 'dist', 'frontend', 'index.html'));

    console.log("Chemin d'index.html:", path.join(__dirname, 'dist', 'frontend', 'browser', 'index.html'));

    mainWindow.loadURL(
      format({
        pathname: path.join(__dirname, 'dist', 'frontend', 'browser', 'index.html'),
        protocol: 'file:',
        slashes: true,
      })
    );
  }

  // Ouvrir le lien externe dans le navigateur par défaut
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' }; // Annule l'ouverture dans Electron
  });

  mainWindow.webContents.session.clearCache().then(() => {
    console.log('Cache réseau effacé');
  });

  mainWindow.setMenuBarVisibility(false); // Masquer le menu pour cette fenêtre

}

app.whenReady().then(() => {
  console.log("Application Electron prête");

  let pythonProcess = null;

  if (app.isPackaged) {
    pythonProcess = spawn(path.join(process.resourcesPath, 'server'));
  }
  else {
    // Démarrage du serveur Python avec l'environnement virtuel
    pythonProcess = spawn('bash', [
      '-c',
      `source ${path.join(__dirname, '../backend/venv/bin/activate')} && python ${path.join(__dirname, '../backend/src/api_server/server.py')}`
    ]);
  }

  globalShortcut.register('CommandOrControl+R', () => {
    console.log('Rechargement désactivé');
  });

  pythonProcess.stdout.on('data', (data) => {
    console.log(`Python server: ${data}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`Python server error: ${data}`);
  });

  // Écoute de la fermeture de l'application pour arrêter le serveur Python proprement
  app.on('before-quit', () => {
    console.log("Arrêt du serveur Python...");
    pythonProcess.kill();
  });

  createWindow();

  // Événements IPC
  ipcMain.on('open-save-dialog', async (event) => {
    const result = await dialog.showSaveDialog({
      title: 'Save the current project',
      defaultPath: 'default_project.cybmasde',
      filters: [{ name: 'CybMASDE Project Files', extensions: ['cybmasde'] }],
    });

    event.sender.send('save-file-response', result.canceled ? null : result.filePath);
  });

  ipcMain.on('open-file-dialog', async (event) => {
    const result = await dialog.showOpenDialog(mainWindow, {
      properties: ['openFile'],
      filters: [{ name: 'All Files', extensions: ['*'] }],
    });

    event.sender.send('selected-file', result.canceled ? null : result.filePaths[0]);
  });

  ipcMain.on('close-app', () => {
    app.quit();
  });

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

// app.commandLine.appendSwitch('no-sandbox');

// app.commandLine.appendSwitch('disable-features', 'OutOfBlinkCors');

// app.commandLine.appendSwitch('disable-dev-shm-usage'); // Utilise la RAM au lieu de /dev/shm

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
