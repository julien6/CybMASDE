import { FormBuilder, Validators, FormsModule, ReactiveFormsModule } from '@angular/forms';
import { ChangeDetectionStrategy, Component, OnInit, inject, computed, signal } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';

export interface Algorithm {
  name: string;
  completed: boolean;
  description?: string;
  subalgorithms?: Algorithm[];
}

@Component({
  selector: 'app-solving',
  templateUrl: './solving.component.html',
  styleUrls: ['./solving.component.css']
})
export class SolvingComponent implements OnInit {

  constructor(private http: HttpClient) { }

  ngOnInit() {
  }

  private _formBuilder = inject(FormBuilder);

  solvingFormGroup = this._formBuilder.group({
    solvingCtrl: ['', Validators.required],
  });

  editorOptions = { theme: 'vs-dark', language: 'python' };

  envSimModel = "";
  runningOutput = "";
  solvingOutput = "";

  readonly algorithm = signal<Algorithm>({
    name: 'All',
    completed: false,
    description: "All algorithms",
    subalgorithms: [
      {
        name: 'Value-Based Algorithms',
        completed: false,
        description: "Algorithms focusing on learning value functions.",
        subalgorithms: [
          {
            name: 'Multi-Agent Q-Learning',
            completed: false,
            description: "A foundational multi-agent extension of Q-learning. \n\n Pros: Simple to implement and understand, suitable for minimal coordination. \n\n Cons: Scalability issues and difficulty handling non-stationarity."
          },
          {
            name: 'MADDPG',
            completed: false,
            description: "An adaptation of DDPG for multi-agent settings. \n\n Pros: Handles continuous action spaces well, enhances coordination. \n\n Cons: Data-intensive and complex to implement."
          }
        ]
      },
      {
        name: 'Policy-Based Algorithms',
        completed: false,
        description: "Algorithms directly learning policies for optimal behavior.",
        subalgorithms: [
          {
            name: 'REINFORCE',
            completed: false,
            description: "A basic policy gradient method for direct policy learning. \n\n Pros: Simple and adaptable to stochastic environments. \n\n Cons: High gradient variance slows convergence."
          },
          {
            name: 'Multi-Agent PPO (MAPPO)',
            completed: true,
            description: "An extension of PPO designed for multi-agent setups. \n\n Pros: Stabilizes policy updates, performs well in various scenarios. \n\n Cons: Requires careful hyperparameter tuning, high computational cost."
          }
        ]
      },
      {
        name: 'Hybrid Algorithms',
        completed: false,
        description: "Combines value-based and policy-based methods.",
        subalgorithms: [
          {
            name: 'A3C (Asynchronous Advantage Actor-Critic)',
            completed: false,
            description: "Mixes policy and value learning for balanced exploration/exploitation. \n\n Pros: Speeds up training through asynchronous execution. \n\n Cons: Needs fine-tuning and complex synchronization."
          },
          {
            name: 'MAPPO',
            completed: false,
            description: "A hybrid that integrates PPO with centralized training. \n\n Pros: Effective for cooperative tasks, good stability. \n\n Cons: Difficult training in highly competitive environments, resource-intensive."
          }
        ]
      },
      {
        name: 'Game-Theoretic and Cooperative Algorithms',
        completed: false,
        description: "Algorithms leveraging game theory for coordination.",
        subalgorithms: [
          {
            name: 'Independent Q-Learning (IQL)',
            completed: false,
            description: "An independent version of Q-learning for each agent. \n\n Pros: Simple and easy to implement. \n\n Cons: Faces severe non-stationarity issues in multi-agent setups."
          },
          {
            name: 'COMA (Counterfactual Multi-Agent Policy Gradients)',
            completed: false,
            description: "Uses counterfactual baselines to address agent contributions. \n\n Pros: Reduces variance and enhances cooperation. \n\n Cons: Computationally demanding due to baseline calculations."
          }
        ]
      },
      {
        name: 'Centralized Training with Decentralized Execution',
        completed: false,
        description: "Trains agents with centralized coordination for decentralized application.",
        subalgorithms: [
          {
            name: 'QMIX',
            completed: false,
            description: "Decomposes Q-values to improve multi-agent coordination. \n\n Pros: Balances centralized training and decentralized action. \n\n Cons: Less effective in highly competitive environments."
          },
          {
            name: 'VDN (Value Decomposition Networks)',
            completed: false,
            description: "Simplifies multi-agent coordination with value decomposition. \n\n Pros: Efficient and simpler than QMIX. \n\n Cons: Limited handling of complex interactions."
          }
        ]
      }
    ]
  });


  automatedHPO = false;

  readonly partiallyComplete = computed(() => {
    const algorithm = this.algorithm();
    if (!algorithm.subalgorithms) {
      return false;
    }
    return algorithm.subalgorithms.some(t => t.completed) && !algorithm.subalgorithms.every(t => t.completed);
  });

  update(completed: boolean, index?: number) {
    this.algorithm.update(algorithm => {
      if (index === undefined) {
        algorithm.completed = completed;
        algorithm.subalgorithms?.forEach(t => (t.completed = completed));
      } else {
        algorithm.subalgorithms![index].completed = completed;
        algorithm.completed = algorithm.subalgorithms?.every(t => t.completed) ?? true;
      }
      return { ...algorithm };
    });
  }

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
    "stoppingInput": {
      'stopModel': { 'fullName': 'Stopping Criteria', 'content': '' },
    },
    "constraintInput": {
      'constraintText': { 'fullName': 'Constraint Text (alpha)', 'content': '' },
      'constraintModel': { 'fullName': 'Constraint Model', 'content': '' },
    }
  }

  outputEnvironmentModel = ""
  selectedFile: File | null = null;
  selectedConstraintInput = "constraintModel"

  onFileSelected(event: Event, inputCategory: string | null = null, inputType: string | null = null): void {
    const input = event.target as HTMLInputElement;

    if (input.files && input.files.length > 0) {
      this.selectedFile = input.files[0];
      if (inputCategory !== null && inputType !== null) {
        const reader = new FileReader();
        reader.onload = () => {
          this.modelingInput[inputCategory!][inputType!]['content'] = reader.result as string;
        };
        reader.readAsText(this.selectedFile);
      }
    }
  }

  openLink(url: string) {
    window.open(url, '_blank');
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

}
