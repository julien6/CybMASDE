import { Component, Input, Output, EventEmitter } from '@angular/core';
import { ConfigEditorService } from '../config-editor.service';
import { ElectronService } from '../electron.service';

@Component({
  selector: 'app-transferring',
  templateUrl: './transfering.component.html',
  styleUrls: ['./transfering.component.scss']
})
export class TransferringComponent {
  @Input() transferring: any;
  @Output() transferringChange = new EventEmitter<any>();

  constructor(
    private editorConfigService: ConfigEditorService,
    private electronService: ElectronService
  ) { }

  /**
   * Update a field in the transferring.configuration object and emit changes upward.
   */
  onValueChange(field: string, value: any) {
    this.editorConfigService
      .setValueByPath(this.transferring.configuration, field, value)
      .then((updatedConfig) => {
        this.transferring.configuration = updatedConfig;
        this.transferringChange.emit(this.transferring);
      });
  }

  /**
   * Open file chooser for selecting a Python API file.
   */
  async onFileSelected(field: string) {
    const filePath = await this.electronService.selectFile();
    this.editorConfigService
      .setValueByPath(this.transferring.configuration, field, filePath as string)
      .then((data) => {
        this.transferring.configuration = data;
        this.transferringChange.emit(this.transferring);
      });
  }
}
