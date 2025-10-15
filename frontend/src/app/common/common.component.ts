import { Component, Input, Output, EventEmitter } from '@angular/core';
import { ConfigEditorService } from '../config-editor.service';
import { ElectronService } from '../electron.service';

@Component({
  selector: 'app-common',
  templateUrl: './common.component.html',
  styleUrls: ['./common.component.css']
})
export class CommonComponent {
  @Input() common: any;
  @Output() commonChange = new EventEmitter<any>();

  constructor(
    private editorConfigService: ConfigEditorService,
    private electronService: ElectronService
  ) { }

  /**
   * Update a field in the common configuration and emit changes upward.
   */
  onValueChange(field: string, value: any) {
    this.editorConfigService
      .setValueByPath(this.common, field, value)
      .then((data) => {
        this.common = data;
        this.commonChange.emit(this.common);
      });
  }

  /**
   * Open file chooser for selecting the label manager Python file.
   */
  async onFileSelected(field: string) {
    const filePath = await this.electronService.selectFile();
    this.editorConfigService
      .setValueByPath(this.common, field, filePath as string)
      .then((data) => {
        this.common = data;
        this.commonChange.emit(this.common);
      });
  }
}
