import { Component, ChangeDetectionStrategy, OnInit, inject } from '@angular/core';
import { FormBuilder, Validators, FormsModule, ReactiveFormsModule } from '@angular/forms';
import { environment } from '../../environments/environment';
import { HttpClient, HttpHeaders } from '@angular/common/http';

@Component({
  selector: 'app-modeling',
  templateUrl: './modeling.component.html',
  styleUrls: ['./modeling.component.css']
})
export class ModelingComponent implements OnInit {

  constructor(private http: HttpClient) { }

  private _formBuilder = inject(FormBuilder);

  editorOptions = { theme: 'vs-dark', language: 'python' };

  selectedEnvironmentInput = "environmentTraces"
  selectedGoalInput = "goalModel"
  selectedConstraintInput = "constraintModel"

  modelingInput: any = {
    "environmentInput": {
      'environmentApi': { 'fullName': 'Environment API URL', 'content': '' },
      'environmentTraces': { 'fullName': 'Environment Traces', 'content': '' },
      'environmentModel': { 'fullName': 'Observation Transition Function Model', 'content': '' }
    },
    "goalInput": {
      'goalText': { 'fullName': 'Goal Text (alpha)', 'content': '' },
      'goalModel': { 'fullName': 'Reward Function Model', 'content': '' },
      'goalStates': { 'fullName': 'Goal States', 'content': '' }
    },
    "constraintInput": {
      'constraintText': { 'fullName': 'Constraint Text (alpha)', 'content': '' },
      'constraintModel': { 'fullName': 'Constraint Model', 'content': '' },
    }
  }

  outputEnvironmentModel = ""
  selectedFile: File | null = null;

  onFileSelected(event: Event, inputCategory: string, inputType: string): void {
    const input = event.target as HTMLInputElement;

    if (input.files && input.files.length > 0) {
      this.selectedFile = input.files[0];
      const reader = new FileReader();
      reader.onload = () => {
        this.modelingInput[inputCategory][inputType]['content'] = reader.result as string;
      };
      reader.readAsText(this.selectedFile);
    }
  }

  // Fonction pour envoyer le fichier au serveur
  uploadFile(inputCategory: string, inputType: string): void {
    if (this.selectedFile) {
      const reader = new FileReader();

      // Lire le contenu du fichier comme texte
      reader.onload = () => {
        const fileContent = reader.result as string;

        try {
          // Convertir le texte en JSON
          const traces = JSON.parse(fileContent);

          // Envoyer les traces au backend
          this.http.post('http://localhost:5001/modeling-transition-traces', traces, {
            headers: new HttpHeaders({
              'Content-Type': 'application/json'
            })
          }).subscribe(
            response => {
              console.log('Réponse du serveur:', response);
            },
            error => {
              console.error('Erreur lors de l\'envoi des traces:', error);
            }
          );
        } catch (error) {
          console.error('Le fichier sélectionné n\'est pas un fichier JSON valide.', error);
        }
      };

      // Lire le fichier
      reader.readAsText(this.selectedFile);
    }
  }

  openLink(url: string) {
    window.open(url, '_blank');
  }

  ngOnInit() {
  }

}
