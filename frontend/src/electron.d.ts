// src/electron.d.ts
interface ElectronAPI {
  ipcRenderer: typeof import('electron').ipcRenderer;
  process: typeof process;
  closeApp: () => void | undefined;
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