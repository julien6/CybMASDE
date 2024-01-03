import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { AngJsoneditorModule } from '@maaxgr/ang-jsoneditor';
import { DetailPanelComponent } from './detail-panel/detail-panel.component'
import { AppNetworkService } from './services/app.network.service';
import { DataService } from './services/app.data.service';
import { NgTerminalModule } from 'ng-terminal';
import { ServerAPIService } from './services/serverAPI.service';
import { NgChartsModule } from 'ng2-charts';
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
import { MenuBarComponent } from './menu-bar/menu-bar.component';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { MatDialogModule } from '@angular/material/dialog';
import { MatExpansionModule } from '@angular/material/expansion';
import { MatDatepickerModule } from '@angular/material/datepicker';
import { MatNativeDateModule } from '@angular/material/core';
import { PreferencesComponent } from './preferences/preferences.component';
import { LibraryComponent } from './library/library.component';
import { PropertiesComponent } from './properties/properties.component';
import { HistoryComponent } from './history/history.component';
import { MatCardModule } from '@angular/material/card';
import { MatGridListModule } from '@angular/material/grid-list';
import { MatTabsModule } from '@angular/material/tabs';
import { MatRippleModule } from '@angular/material/core';
import { LibraryElementComponent } from './library-element/library-element.component'
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { ConfigureRunComponent } from './configure-run/configure-run.component';
import { PackagingComponent } from './packaging/packaging.component';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { MatInputModule } from '@angular/material/input';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatStepperModule } from '@angular/material/stepper';
import { ConfigShareService } from './services/config-share.service';
import { NgFor } from '@angular/common';
import { MatSelectModule } from '@angular/material/select';
import { EditorModule } from './editor/editor.module';
import { Mpld3Component } from './mpld3/mpld3.component';
import { PixelVisualizationModule } from './pixel-visualization/pixel-visualization.module';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { AgentsLogsComponent } from './agents-logs/agents-logs.component';
import { AgentsPixelsComponent } from './agents-pixels/agents-pixels.component';
import { AgentsHistogramsComponent } from './agents-histograms/agents-histograms.component';
import { AgentsPiechartsComponent } from './agents-piecharts/agents-piecharts.component';
import { AgentsSeqdiagramsComponent } from './agents-seqdiagrams/agents-seqdiagrams.component';
import { NetworkNodesComponent } from './network-nodes/network-nodes.component';
import { NetworkTopologyComponent } from './network-topology/network-topology.component';
import { GlobalGraphsComponent } from './global-graphs/global-graphs.component';
import { GlobalMatrixesComponent } from './global-matrixes/global-matrixes.component';
import { GlobalSeqdiagramsComponent } from './global-seqdiagrams/global-seqdiagrams.component';
import { MatTableModule } from '@angular/material/table'
import { MarkdownModule, MarkedOptions, MarkedRenderer } from 'ngx-markdown';
import { MatSortModule } from '@angular/material/sort';
import { MatTreeModule } from '@angular/material/tree';
import { MatSliderModule } from '@angular/material/slider';
import { TreeModule } from './tree/tree.module';
import { Mpld3Module } from './mpld3/mpld3.module';

@NgModule({
  declarations: [
    AppComponent,
    DetailPanelComponent,
    AppComponent,
    MenuBarComponent,
    PreferencesComponent,
    LibraryComponent,
    PropertiesComponent,
    HistoryComponent,
    LibraryElementComponent,
    ConfigureRunComponent,
    PackagingComponent,
    AgentsLogsComponent,
    AgentsPixelsComponent,
    AgentsHistogramsComponent,
    AgentsPiechartsComponent,
    AgentsSeqdiagramsComponent,
    NetworkNodesComponent,
    NetworkTopologyComponent,
    GlobalGraphsComponent,
    GlobalMatrixesComponent,
    GlobalSeqdiagramsComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    AngJsoneditorModule,
    MatSliderModule,
    MatButtonModule,
    MatTreeModule,
    MatIconModule,
    MatFormFieldModule,
    MatSortModule,
    MatSelectModule,
    NgFor,
    NgTerminalModule,
    MatTableModule,
    EditorModule,
    Mpld3Module,
    PixelVisualizationModule,
    NgChartsModule,
    MatProgressBarModule,
    TreeModule,
    NoopAnimationsModule,
    MatStepperModule,
    ReactiveFormsModule,
    BrowserAnimationsModule,
    HttpClientModule,
    MarkdownModule.forRoot({
      loader: HttpClient,
      markedOptions: {
        provide: MarkedOptions,
        useFactory: markedOptionsFactory
      }
    }),
    MatTabsModule,
    MatRippleModule,
    MatSlideToggleModule,
    LayoutModule,
    MatToolbarModule,
    FormsModule,
    MatButtonModule,
    MatSidenavModule,
    MatIconModule,
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
  providers: [AppNetworkService, DataService, ConfigShareService, ServerAPIService, HttpClient],
  bootstrap: [AppComponent]
})
export class AppModule { }

// function that returns `MarkedOptions` with renderer override
export function markedOptionsFactory(): MarkedOptions {
  const renderer = new MarkedRenderer();
  renderer.code = function (code, language: any) {
    if (language.match(/^mermaid/)) {
      return '<div class="mermaid">' + code + '</div>';
    } else {
      return '<pre><code>' + code + '</code></pre>';
    }
  };
  return {
    renderer: renderer,
    gfm: true,
    breaks: false,
    pedantic: false,
    smartLists: true,
    smartypants: false
  };
}