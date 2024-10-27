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
  closeApp: () => ipcRenderer.send('close-app')
});