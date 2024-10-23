import { Component, ElementRef, ViewChild } from '@angular/core';

@Component({
  selector: 'app-menu-bar',
  templateUrl: './menu-bar.component.html',
  styleUrls: ['./menu-bar.component.css']
})
export class MenuBarComponent {

  @ViewChild('fileInput', { static: false })
  fileInput!: ElementRef;

  constructor() { }

  ngOnInit() {
  }

  onWork = false;

  createNewProject() {
    // TODO: créer un nouveau projet en backend et envoyer dans l'interface
    this.onWork = true
  }

  // Fonction pour déclencher l'ouverture de l'explorateur de fichiers
  openFileDialog() {
    this.fileInput.nativeElement.click();
  }

  // Fonction pour gérer la sélection du fichier
  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;

    if (input.files && input.files.length > 0) {
      const file = input.files[0];

      // Vérifier que le fichier est bien un fichier JSON
      if (file.type === 'application/json') {
        const reader = new FileReader();

        // Lire le fichier comme texte
        reader.onload = () => {
          const fileContent = reader.result as string;
          try {
            // Convertir le texte en objet JSON
            const json = JSON.parse(fileContent);
            console.log('JSON project file loaded:', json);
            // TODO: load the JSON file into the interface
            this.onWork = true
            // Ici, vous pouvez traiter le contenu JSON comme vous le souhaitez
          } catch (error) {
            console.error('Error while loading the JSON project file:', error);
          }
        };

        reader.readAsText(file);
      } else {
        console.error('Please select a JSON project file.');
      }
    }
  }

}
