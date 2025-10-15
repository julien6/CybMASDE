import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { NoopAnimationsModule } from '@angular/platform-browser/animations';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { MatSlideToggleModule } from '@angular/material/slide-toggle';
import { LayoutModule } from '@angular/cdk/layout';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatButtonModule } from '@angular/material/button';
import { MatSidenavModule } from '@angular/material/sidenav';
import { MatIconModule } from '@angular/material/icon';
import { MatListModule } from '@angular/material/list';
import { MatMenuModule } from '@angular/material/menu';
import { ActivitiesComponent } from './activities/activities.component';
import { ModelingComponent } from './modeling/modeling.component';
import { RefiningComponent } from './refining/refining.component';
import { TrainingComponent } from './training/training.component';
import { AnalyzingComponent } from './analyzing/analyzing.component';
import { CommonComponent } from './common/common.component';
import { TransferringComponent } from './transfering/transfering.component';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { MatDialogModule } from '@angular/material/dialog';
import { MatExpansionModule } from '@angular/material/expansion';
import { MatDatepickerModule } from '@angular/material/datepicker';
import { MatNativeDateModule } from '@angular/material/core';
import { MatCardModule } from '@angular/material/card';
import { MatGridListModule } from '@angular/material/grid-list';
import { MatTabsModule } from '@angular/material/tabs';
import { MatRippleModule } from '@angular/material/core';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { MatInputModule } from '@angular/material/input';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatStepperModule } from '@angular/material/stepper';
import { NgFor } from '@angular/common';
import { MatSelectModule } from '@angular/material/select';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatTableModule } from '@angular/material/table'
import { MatSortModule } from '@angular/material/sort';
import { MatTreeModule } from '@angular/material/tree';
import { MatSliderModule } from '@angular/material/slider';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MonacoEditorModule } from 'ngx-monaco-editor-v2';
import { MatChipsModule } from '@angular/material/chips';
import { MenuBarComponent } from './menu-bar/menu-bar.component';
import { HomeComponent } from './home/home.component';
import { AboutDialogComponent } from './about-dialog/about-dialog.component';
import { MatTooltipHtmlDirective } from './directives/mat-tooltip-html.directive';

@NgModule({
  declarations: [
    AppComponent,
    ActivitiesComponent,
    ModelingComponent,
    TrainingComponent,
    RefiningComponent,
    HomeComponent,
    CommonComponent,
    AnalyzingComponent,
    AboutDialogComponent,
    TransferringComponent,
    MenuBarComponent,
    MatTooltipHtmlDirective
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    MonacoEditorModule.forRoot(),
    MatSliderModule,
    MatTooltipModule,
    MatTreeModule,
    MatSortModule,
    MatChipsModule,
    MatSelectModule,
    NgFor,
    MatTableModule,
    MatProgressBarModule,
    NoopAnimationsModule,
    MatStepperModule,
    ReactiveFormsModule,
    BrowserAnimationsModule,
    HttpClientModule,
    MatTabsModule,
    MatRippleModule,
    MatSlideToggleModule,
    LayoutModule,
    MatToolbarModule,
    FormsModule,
    MatSidenavModule,
    MatCardModule,
    MatListModule,
    MatGridListModule,
    MatMenuModule,
    MatCheckboxModule,
    MatDialogModule,
    MatButtonModule,
    MatExpansionModule,
    MatIconModule,
    MatFormFieldModule,
    MatInputModule,
    MatDatepickerModule,
    MatNativeDateModule
  ],
  providers: [HttpClient],
  bootstrap: [AppComponent]
})
export class AppModule { }
