import { Component, Input, Output, EventEmitter } from '@angular/core';
import { ModellingConfig } from '../models/config.model';
import { ConfigEditorService } from '../config-editor.service';
import { ElectronService } from '../electron.service';

@Component({
  selector: 'app-modeling',
  templateUrl: './modeling.component.html',
  styleUrls: ['./modeling.component.scss']
})
export class ModelingComponent {
  @Input() model!: ModellingConfig;
  @Output() modelChange = new EventEmitter<ModellingConfig>();

  constructor(private editorConfigService: ConfigEditorService, private electronService: ElectronService) { }

  jsonEditorOptions = { theme: 'vs-dark', language: 'json', automaticLayout: true };
  selectedFile: File | null = null;
  selectedFilePath: string | null = null;

  // Local buffers for Monaco editors
  autoencoderJson = '';
  rdlmJson = '';

  ngOnInit() {
    // Initialize editor contents from model
    if (this.model?.generated_environment?.world_model?.jopm) {
      this.autoencoderJson = JSON.stringify(
        this.model.generated_environment.world_model.jopm.autoencoder.hyperparameters || {},
        null,
        2
      );
      this.rdlmJson = JSON.stringify(
        this.model.generated_environment.world_model.jopm.rdlm.hyperparameters || {},
        null,
        2
      );
    }
  }

  /** Called when user edits the JSON directly in the editor */
  onJsonChange(field: 'autoencoder' | 'rdlm', value: string) {
    try {
      const parsed = JSON.parse(value);
      if (field === 'autoencoder') {
        this.model.generated_environment.world_model.jopm.autoencoder.hyperparameters = parsed;
      } else {
        this.model.generated_environment.world_model.jopm.rdlm.hyperparameters = parsed;
      }
      this.modelChange.emit(this.model);
    } catch {
      // Invalid JSON, do not overwrite model yet
    }
  }

  async onFileSelected(field_path: string) {
    const filePath = await this.electronService.selectFile();
    this.editorConfigService.setValueByPath(this.model, field_path, filePath as string).then((data) => {
      this.model = data;
      this.modelChange.emit(this.model);
    });
  }

  /**
   * Safe JSON parse helper for use from template expressions.
   * Returns parsed object or an empty object on error.
   */
  parseJson(value: string): any {
    try {
      return value && value.trim() ? JSON.parse(value) : {};
    } catch (e) {
      // swallow parse errors and return an empty object to keep template stable
      return {};
    }
  }

  /**
   * File chooser for JSON hyperparameter files.
   * Reads, parses, and assigns content directly into the bound object.
   */
  openFileContent(path: string, field: 'autoencoder' | 'rdlm'): void {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json,application/json';
    input.onchange = async (event: any) => {
      const file = event.target.files[0];
      if (file) {
        try {
          const text = await file.text();
          const parsed = JSON.parse(text);
          this.editorConfigService.setValueByPath(this.model, path, parsed).then((data) => {
            this.model = data;
            this.modelChange.emit(this.model);
          });
          // Update editor text
          if (field === 'autoencoder') this.autoencoderJson = JSON.stringify(parsed, null, 2);
          if (field === 'rdlm') this.rdlmJson = JSON.stringify(parsed, null, 2);
        } catch (error) {
          console.error(`❌ Error reading JSON file for ${path}:`, error);
          alert(`Failed to read or parse JSON file: ${file.name}`);
        }
      }
    };
    input.click();
  }

}
