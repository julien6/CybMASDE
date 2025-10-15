import { Injectable } from '@angular/core';
import { ProjectConfig } from './models/config.model';
import { HttpClient } from '@angular/common/http';
import { ElectronService } from './electron.service';
import { AboutDialogComponent } from './about-dialog/about-dialog.component';
import { MatDialog } from '@angular/material/dialog';
import { BehaviorSubject } from 'rxjs';


@Injectable({ providedIn: 'root' })
export class ConfigEditorService {

  constructor(private http: HttpClient, private electronService: ElectronService) { }

  rootUrl = "http://127.0.0.1:5001/";

  ngOnInit() {
  }

  private projectConfig = new BehaviorSubject<ProjectConfig | null>(null);
  private onWork = new BehaviorSubject<boolean>(false);
  private saved = new BehaviorSubject<boolean>(true);

  // Observables pour écoute réactive (si besoin)
  onWork$ = this.onWork.asObservable();
  saved$ = this.saved.asObservable();
  // Getter direct (synchronisé)
  get isOnWork(): boolean {
    return this.onWork.value;
  }
  get isSaved(): boolean {
    return this.saved.value;
  }
  // Setter global
  setOnWork(state: boolean): void {
    this.onWork.next(state);
  }
  setSaved(state: boolean): void {
    this.saved.next(state);
  }

  // Observable pour écoute réactive (si besoin)
  config$ = this.projectConfig.asObservable();

  // Getter direct (synchronisé)
  get currentConfig(): ProjectConfig | null {
    return this.projectConfig.value;
  }

  // Setter global
  setConfig(newConfig: ProjectConfig): void {
    this.projectConfig.next(newConfig);
  }

  // Mise à jour partielle
  patch(partial: Partial<ProjectConfig>): void {
    const current = this.currentConfig;
    if (current) {
      const updated = { ...current, ...partial };
      this.projectConfig.next(updated);
    }
  }

  // Mise à jour par chemin (ex: "modeling.mesh_file")
  setValueByPath(obj: any, path: string, value: any): Promise<any> {

    return new Promise<any>((resolve, reject) => {

      if (!obj || !path) return reject("Invalid object or path");

      // Deep clone to trigger Angular change detection
      const updatedObj = JSON.parse(JSON.stringify(obj));
      const keys = path.split('.');
      let current = updatedObj;

      for (let i = 0; i < keys.length - 1; i++) {
        const key = keys[i];
        if (!current[key] || typeof current[key] !== 'object') {
          current[key] = {};
        }
        current = current[key];
      }

      current[keys[keys.length - 1]] = value;
      resolve(updatedObj);

    });

  }

  // Mise à jour par chemin (ex: "modeling.mesh_file")
  setConfigurationValueByPath(path: string, value: any): void {
    this.setValueByPath(this.projectConfig.value, path, value);
  }

  getRecentProjects(): Promise<any> {
    return new Promise<any>((resolve, reject) => {
      this.http.get(this.rootUrl + 'get-recent-projects').subscribe(data => {
        resolve(data);
      }, error => {
        console.error('Erreur lors du chargement des projets récents', error);
        reject(error);
      });
    });
  }

  newProject(): Promise<any> {
    return new Promise<any>((resolve, reject) => {
      this.http.get(this.rootUrl + 'new-project').subscribe(data => {
        this.projectConfig.next(data ? (data as ProjectConfig) : {} as ProjectConfig);
        this.setOnWork(true);
        this.setSaved(false);
        resolve(data);
      }, error => {
        console.error('Erreur lors de la création d\'un nouveau projet', error);
        reject(error);
      });
    });
  }

  loadProject(filePath: string): Promise<ProjectConfig> {
    return new Promise<ProjectConfig>((resolve, reject) => {
      this.http.get(this.rootUrl + 'load-project?path=' + filePath).subscribe(data => {
        this.setOnWork(true);
        this.setSaved(false);
        this.projectConfig.next(data ? (data as ProjectConfig) : {} as ProjectConfig);
        resolve(data ? (data as ProjectConfig) : {} as ProjectConfig);
      }, error => {
        console.error('Error while loading project', error);
        reject(error);
      });
    });
  }

  saveProjectAs(filePath: string | null): Promise<any> {
    return new Promise<any>((resolve, reject) => {
      if (filePath) {
        this.http.post(this.rootUrl + 'save-project-as?path=' + filePath, JSON.stringify(this.projectConfig.value)).subscribe(data => {
          this.setSaved(true);
          resolve(data);
        }, error => {
          console.error('Error while loading non saved project', error);
          reject(error);
        });
      } else {
        console.log('Sauvegarde annulée');
        reject('Sauvegarde annulée');
      }
    });
  }

  saveProject(): Promise<any> {
    return new Promise<any>((resolve, reject) => {
      this.http.post(this.rootUrl + 'save-project', JSON.stringify(this.projectConfig.value)).subscribe(data => {
        this.setSaved(true);
        resolve(data);
      }, error => {
        console.error('Error while saving project', error);
        reject(error);
      });
    });
  }

  saveAndRun(): Promise<any> {
    return new Promise<any>((resolve, reject) => {
      this.http.post(
        this.rootUrl + 'save-and-run',
        JSON.stringify(this.projectConfig.value),
        { headers: { 'Content-Type': 'application/json' } }
      ).subscribe({
        next: (data) => {
          this.setSaved(true);
          resolve(data);
        },
        error: (error) => {
          console.error('Error while saving and running project', error);
          reject(error);
        }
      });
    });
  }

  openProjectDialog(): Promise<ProjectConfig | null> {
    return new Promise<ProjectConfig | null>((resolve, reject) => {
      // Ouvre le sélecteur de fichiers via Electron
      window["electron"].openFileDialog();
      // Récupère le chemin du fichier sélectionné
      window["electron"].onFileSelected((path: string) => {
        this.loadProject(path).then((config) => {
          resolve(config);
        }).catch((error) => {
          console.error('Error while loading project', error);
          reject(error);
        });
      });
    });
  }

  saveProjectAsDialog(): Promise<any> {
    return new Promise<any>(async (resolve, reject) => {
      // Ouvre le selecteur de fichiers via Electron et récupère le chemin du fichier sélectionné
      const savedFilePath: string | null = await this.electronService.saveFile();
      this.saveProjectAs(savedFilePath).then((data) => {
        resolve(data);
      }).catch((error) => {
        console.error('Error while loading non saved project', error);
        reject(error);
      });
    });
  }

}
