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
  selectedStoppingInput = "stopModel"
  selectedRenderingInput = "renderModel"

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
    },
    "stoppingInput": {
      'stopModel': { 'fullName': 'Stopping Criteria', 'content': '' }
    },
    "renderingInput": {
      'renderModel': { 'fullName': 'Rendering Function Model', 'content': '' }
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
    const d = `
class ComponentFunctions:
    def __init__(self, label_manager=None):
        self.label_manager = label_manager
        self.iteration = 0

    def reward_fn(self, current_obs, action, next_obs):
        # Example implementation of reward function
        rewards = {}
        for agent_id in current_obs:
            # Here you would implement your logic to compute the reward
            rewards[agent_id] = 0.5 + random.random(
            ) * self.iteration  # Placeholder value
        return rewards

    def done_fn(self, current_obs, action, next_obs):
        # Example implementation of done function
        dones = {}
        for agent_id in current_obs:
            # Placeholder value
            dones[agent_id] = False if self.iteration < 21 else True
        self.iteration = 0 if self.iteration >= 21 else self.iteration + 1
        return dones

    def render_fn(self, current_obs, action, next_obs):
        # Example implementation of render function
        return None  # Placeholder for rendering logic

    `;

    this.modelingInput["goalInput"]["goalModel"]['content'] = d;
    this.modelingInput["stoppingInput"]["stopModel"]['content'] = d;
    this.modelingInput["renderingInput"]["renderModel"]['content'] = d;
  }


}
