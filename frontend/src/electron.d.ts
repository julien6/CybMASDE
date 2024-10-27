// src/electron.d.ts
interface ElectronAPI {
  ipcRenderer: typeof import('electron').ipcRenderer;
  process: typeof process;
  closeApp: () => void | undefined;
  openFileDialog: () => void;  // Nouvelle méthode pour ouvrir la boîte de dialogue
  onFileSelected: (callback: (path: string) => void) => void;  // Nouvelle méthode pour récupérer le chemin du fichier
}

interface Window {
  electron: ElectronAPI;
}

declare var window: Window;

// const { contextBridge, ipcRenderer } = require('electron');

// contextBridge.exposeInMainWorld('electron', {
//   ipcRenderer: ipcRenderer,
//   process: {
//     versions: process.versions,
//   },
// });