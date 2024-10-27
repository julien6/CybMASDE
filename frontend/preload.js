const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electron', {
  ipcRenderer: {
    // Expose uniquement les méthodes nécessaires pour éviter les problèmes de sécurité
    send: (channel, data) => ipcRenderer.send(channel, data),
    once: (channel, func) =>
      ipcRenderer.once(channel, (event, ...args) => func(event, ...args)),
  },
  process: {
    versions: process.versions,
  },
  closeApp: () => ipcRenderer.send('close-app'),
  openFileDialog: () => ipcRenderer.send('open-file-dialog'),
  onFileSelected: (callback) => ipcRenderer.once('selected-file', (event, path) => callback(path)),
});
