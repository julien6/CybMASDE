import { Component, ElementRef, ViewChild } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { ElectronService } from '../electron.service';


@Component({
  selector: 'app-menu-bar',
  templateUrl: './menu-bar.component.html',
  styleUrls: ['./menu-bar.component.css']
})
export class MenuBarComponent {

  @ViewChild('fileInput', { static: false })
  fileInput!: ElementRef;

  constructor(private http: HttpClient, private electronService: ElectronService) { }

  ngOnInit() {
  }

  rootUrl = "http://127.0.0.1:5000/";
  onWork = false;

  createNewProject() {

    this.http.get(this.rootUrl + 'new-project').subscribe(data => {
      this.onWork = true;
    }, error => {
      console.error('Erreur lors du chargement des projets récents', error);
    });

  }

  savedFilePath: string | null = null;

  async saveProjectAs() {
    this.savedFilePath = await this.electronService.saveFile();
    if (this.savedFilePath) {
      this.http.get(this.rootUrl + 'save-project-as?path=' + this.savedFilePath).subscribe(data => {
        // console.log(data);
        this.onWork = true;
      }, error => {
        console.error('Error while loading non saved project', error);
      });

    } else {
      console.log('Sauvegarde annulée');
    }
  }

  saveProject() {

    this.http.get(this.rootUrl + 'save-project').subscribe(data => {
    }, error => {
      console.error('Error while saving project', error);
    });

  }

  openProject() {
    this.openFileDialog()
  }

  // Fonction pour déclencher l'ouverture de l'explorateur de fichiers
  openFileDialog() {
    this.fileInput.nativeElement.click();
  }

  closeApp() {
    if (window.electron && window["electron"].closeApp !== null) {
      window["electron"].closeApp();
    } else {
      console.warn("La fonction closeApp n'est pas disponible.");
    }
  }

}
