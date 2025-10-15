import { Component, Input, OnInit, Output, EventEmitter } from '@angular/core';
import { ConfigEditorService } from '../config-editor.service';
import { ElectronService } from '../electron.service';

@Component({
  selector: 'app-training',
  templateUrl: './training.component.html',
  styleUrls: ['./training.component.scss']
})
export class TrainingComponent implements OnInit {
  @Input() training: any;
  @Output() trainingChange = new EventEmitter<any>();

  jsonEditorOptions = {
    theme: 'vs-dark',
    language: 'json',
    automaticLayout: true,
    minimap: { enabled: false }
  };

  // Local buffers for JSON editors
  hyperparametersJson = '';
  statisticsJson = '';

  constructor(
    private editorConfigService: ConfigEditorService,
    private electronService: ElectronService
  ) { }

  ngOnInit() {
    // Initialize editors with existing training data (if already loaded)
    this.hyperparametersJson = JSON.stringify(
      this.training?.hyperparameters || {},
      null,
      2
    );
    this.statisticsJson = JSON.stringify(
      this.training?.statistics || {},
      null,
      2
    );
  }

  /**
   * When user edits JSON directly in the editor.
   * We only update the training if valid JSON.
   */
  onJsonChange(field: 'hyperparameters' | 'statistics', value: string) {
    try {
      const parsed = JSON.parse(value);
      this.training[field] = parsed;
      this.trainingChange.emit(this.training);
    } catch {
      // invalid JSON, ignore
    }
  }

  /**
   * Opens a file picker to select a path (for folders or non-JSON files).
   */
  async onFileSelected(field_path: string) {
    const filePath = await this.electronService.selectFile();
    this.editorConfigService
      .setValueByPath(this.training, field_path, filePath as string)
      .then((data) => (this.training = data));
    // emit change so parent two-way binding updates
    this.trainingChange.emit(this.training);
  }

  /**
   * Opens and reads a JSON file, assigns its parsed content into training[field],
   * and updates the associated editor.
   */
  openFileContent(field: 'hyperparameters' | 'statistics') {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json,application/json';
    input.onchange = async (event: any) => {
      const file = event.target.files[0];
      if (file) {
        try {
          const text = await file.text();
          const parsed = JSON.parse(text);
          this.training[field] = parsed;

          this.editorConfigService.setValueByPath(this.training, field, parsed).then((data) => {
            this.training = data;
            this.trainingChange.emit(this.training);
          });

          if (field === 'hyperparameters')
            this.hyperparametersJson = JSON.stringify(parsed, null, 2);
          else this.statisticsJson = JSON.stringify(parsed, null, 2);
          // propagate change to parent
          this.trainingChange.emit(this.training);
        } catch (err) {
          console.error(`❌ Error reading JSON file for ${field}:`, err);
          alert(`Failed to read or parse JSON file: ${file.name}`);
        }
      }
    };
    input.click();
  }


}
