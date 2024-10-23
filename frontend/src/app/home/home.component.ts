import { Component, Optional } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { MenuBarComponent } from '../menu-bar/menu-bar.component';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css']
})
export class HomeComponent {

  recentProjects: string[] = [];

  constructor(private http: HttpClient, @Optional() private menuBar: MenuBarComponent) {}

  ngOnInit(): void {
    this.loadRecentProjects();
  }

  // Fonction pour charger les projets récents depuis un fichier .txt
  loadRecentProjects() {
    // Suppose que le fichier recent_files_history.txt est accessible sur le serveur
    this.http.get('../assets/recent_files_history.txt', { responseType: 'text' }).subscribe(data => {
      this.recentProjects = data.split('\n').filter(line => line.trim() !== '');
    }, error => {
      console.error('Erreur lors du chargement des projets récents', error);
    });
  }

  // Action pour créer un nouveau projet
  newProject() {
    this.menuBar.createNewProject();
  }

  // Action pour charger un projet
  loadProject() {
    this.menuBar.openFileDialog();
  }

}
