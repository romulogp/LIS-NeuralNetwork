package br.com.rgp.neuralnetwork.model;

public enum LisFlower {

    SETOSA("Sitosa"),
    VERSICOLOR("Versicolor"),
    VIRGINIA("Virginia");
    
    private final String descricao;
    
    private LisFlower(String descricao) {
        this.descricao = descricao;
    }
    
    public String getDescricao() {
        return descricao;
    }
    
}
